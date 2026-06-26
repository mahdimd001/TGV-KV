import torch
import math
import logging

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.ERROR)
logger = logging.getLogger(__name__)

class FormanRicciTensorGPU:
    def __init__(self, adj_matrix: torch.Tensor, method="augmented", batch_size=1024):
        """
        A purely GPU-accelerated class to compute Forman-Ricci curvature.
        
        Parameters
        ----------
        adj_matrix : torch.Tensor
            (N, N) dense adjacency matrix. Must be on the target device (e.g., 'cuda').
        method : {"1d", "augmented"}
            Computation method.
        batch_size : int
            Number of edges to process in parallel on the GPU.
        """
        self.device = adj_matrix.device
        self.method = method
        self.batch_size = batch_size
        
        # Enforce float32 and symmetry (undirected graph assumption)
        self.adj_matrix = adj_matrix.float()
        self.adj_matrix = torch.max(self.adj_matrix, self.adj_matrix.t()) 
        self.num_nodes = self.adj_matrix.shape[0]

        self._preprocess_tensors()

    def _preprocess_tensors(self):
        """Extracts edge indices and weights directly from the adjacency matrix on the GPU."""
        #logger.info(f"Preprocessing tensor graph on {self.device}...")

        # 1. Default Node Weights to 1.0
        self.node_weights = torch.ones(self.num_nodes, dtype=torch.float32, device=self.device)

        # 2. Extract Edge List and Weights natively on GPU
        # nonzero() returns (E, 2) where entries are > 0, transpose to get (2, E)
        self.edge_index = self.adj_matrix.nonzero(as_tuple=False).t() 
        self.num_edges = self.edge_index.shape[1]
        
        # Extract flat edge weights using the extracted indices
        self.edge_weights_flat = self.adj_matrix[self.edge_index[0], self.edge_index[1]]

    def compute_ricci_curvature(self):
        """Compute Forman-Ricci curvature entirely on GPU and return tensors."""
        edge_curvatures = torch.zeros(self.num_edges, device=self.device)
        
        #logger.info(f"Starting {self.method} computation on {self.num_edges} edges...")

        # --- EDGE CURVATURE LOOP (BATCHED) ---
        num_batches = math.ceil(self.num_edges / self.batch_size)
        
        for i in range(num_batches):
            start_idx = i * self.batch_size
            end_idx = min((i + 1) * self.batch_size, self.num_edges)
            
            batch_indices = torch.arange(start_idx, end_idx, device=self.device)
            src_idx = self.edge_index[0, batch_indices]
            dst_idx = self.edge_index[1, batch_indices]
            
            w_e = self.edge_weights_flat[batch_indices]
            w_v1 = self.node_weights[src_idx]
            w_v2 = self.node_weights[dst_idx]

            if self.method == "1d":
                curv = self._compute_batch_1d(src_idx, dst_idx, w_e, w_v1, w_v2)
            elif self.method == "augmented":
                curv = self._compute_batch_augmented(src_idx, dst_idx, w_e, w_v1, w_v2)
            else:
                raise ValueError(f"Unknown method: {self.method}")
            
            edge_curvatures[batch_indices] = curv
            
        # --- NODE CURVATURE CALCULATION ---
        #logger.info("Computing node curvatures...")
        node_curv_sum = torch.zeros(self.num_nodes, device=self.device)
        degrees = torch.zeros(self.num_nodes, device=self.device)
        
        # Note: Because nonzero() on a symmetric matrix yields both (u,v) and (v,u),
        # we only need to scatter_add over src_idx (edge_index[0]) to avoid double-counting.
        node_curv_sum.index_add_(0, self.edge_index[0], edge_curvatures)
        degrees.index_add_(0, self.edge_index[0], torch.ones_like(edge_curvatures))
        
        degrees[degrees == 0] = 1.0
        node_curvatures = node_curv_sum / degrees
        
        #logger.info("Computation complete.")
        
        # Return the results as raw tensors. 
        # You can map edge_curvatures back to an NxN matrix if needed.
        return self.edge_index, edge_curvatures, node_curvatures

    def _compute_batch_1d(self, src, dst, w_e, w_v1, w_v2):
        src_rows = torch.index_select(self.adj_matrix, 0, src) 
        dst_rows = torch.index_select(self.adj_matrix, 0, dst)

        src_rows.scatter_(1, dst.unsqueeze(1), 0)
        dst_rows.scatter_(1, src.unsqueeze(1), 0)

        w_e_expanded = w_e.unsqueeze(1)
        epsilon = 1e-8
        
        denom_v1 = torch.sqrt(w_e_expanded * src_rows)
        term_v1 = (w_v1.unsqueeze(1) / (denom_v1 + epsilon))
        term_v1 = term_v1 * (src_rows > 0).float()
        ev1_sum = torch.sum(term_v1, dim=1)

        denom_v2 = torch.sqrt(w_e_expanded * dst_rows)
        term_v2 = (w_v2.unsqueeze(1) / (denom_v2 + epsilon))
        term_v2 = term_v2 * (dst_rows > 0).float()
        ev2_sum = torch.sum(term_v2, dim=1)

        return w_e * ( (w_v1 / w_e) + (w_v2 / w_e) - (ev1_sum + ev2_sum) )

    def _compute_batch_augmented(self, src, dst, w_e, w_v1, w_v2):
        src_rows = torch.index_select(self.adj_matrix, 0, src) 
        dst_rows = torch.index_select(self.adj_matrix, 0, dst)

        src_mask = (src_rows > 0).float()
        dst_mask = (dst_rows > 0).float()
        face_mask = src_mask * dst_mask 

        src_mask.scatter_(1, dst.unsqueeze(1), 0) 
        dst_mask.scatter_(1, src.unsqueeze(1), 0)
        
        v1_only_mask = src_mask - face_mask 
        v2_only_mask = dst_mask - face_mask

        w_f = 1.0 
        num_faces = torch.sum(face_mask, dim=1)
        sum_ef = (w_e / w_f) * num_faces
        
        sum_ve = (w_v1 / w_e) + (w_v2 / w_e)
        sum_ehef = 0.0

        w_e_expanded = w_e.unsqueeze(1)
        epsilon = 1e-8

        denom_v1 = torch.sqrt(w_e_expanded * src_rows) + epsilon
        term_v1 = (w_v1.unsqueeze(1) / denom_v1) * v1_only_mask 
        
        denom_v2 = torch.sqrt(w_e_expanded * dst_rows) + epsilon
        term_v2 = (w_v2.unsqueeze(1) / denom_v2) * v2_only_mask

        sum_veeh = torch.sum(term_v1, dim=1) + torch.sum(term_v2, dim=1)

        return w_e * (sum_ef + sum_ve - torch.abs(sum_ehef - sum_veeh))

