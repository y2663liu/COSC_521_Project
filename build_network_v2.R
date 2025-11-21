# build_interval_networks_v2.R
library(igraph)

# ---------------------- Configuration ----------------------
CFG <- list(
  base_dir          = "./../PressureSensorPi/",
  participant       = "Mark_Right", 
  gesture           = "Press",       
  iou_threshold     = 0.20,           # IoU to determine if two blobs are the "same object"
  movement_threshold= 0.5,            # Minimum pixel shift required to draw edges (Filters stationary presses)
  min_seq_len       = 10,             # Minimum frames required for a valid sequence
  out_dir           = "networks_flow",
  plot_png          = TRUE,
  png_width         = 1200,
  png_height        = 900
)

# ---------------------- Utilities ----------------------
msg <- function(...) cat(sprintf(...), "\n")

# 2x2 Max Pooling (Preserved from your original code)
pool_2x2 <- function(mat) {
  nr <- nrow(mat); nc <- ncol(mat)
  if (is.null(nr) || nr < 2 || nc < 2) return(mat)
  r1 <- seq(1, nr - 1, by = 2); c1 <- seq(1, nc - 1, by = 2)
  pooled <- matrix(0, nrow = length(r1), ncol = length(c1))
  for (i in seq_along(r1)) {
    for (j in seq_along(c1)) {
      pooled[i, j] <- max(mat[r1[i]:(r1[i]+1), c1[j]:(c1[j]+1)], na.rm = TRUE)
    }
  }
  pooled
}

# Read single frame
read_frame_matrix <- function(base_dir, participant, gesture, frame_idx) {
  fn <- file.path(base_dir, "harsh_process", participant, gesture, sprintf("f%05d.csv", frame_idx))
  if (!file.exists(fn)) return(NULL)
  tryCatch({
    mat <- as.matrix(utils::read.csv(fn, header = FALSE, check.names = FALSE))
    pool_2x2(mat) 
  }, error = function(e) NULL)
}

# ---------------------- Advanced Processing Logic ----------------------

# 1. Load, Filter, and Interpolate Sequence
load_and_clean_sequence <- function(cfg, start_f, end_f) {
  # Load all frames into a list
  frames <- list()
  indices <- start_f:end_f
  
  # Check Length (Requirement 2)
  if (length(indices) < cfg$min_seq_len) {
    return(list(valid = FALSE, reason = "Sequence too short"))
  }
  
  raw_mats <- vector("list", length(indices))
  has_data <- logical(length(indices))
  
  for (i in seq_along(indices)) {
    m <- read_frame_matrix(cfg$base_dir, cfg$participant, cfg$gesture, indices[i])
    if (!is.null(m)) {
      raw_mats[[i]] <- m
      if (sum(m, na.rm=TRUE) > 0) has_data[i] <- TRUE
    }
  }
  
  # Check for Empty Sequence (Requirement 2)
  if (!any(has_data)) {
    return(list(valid = FALSE, reason = "Sequence contains no pressure data"))
  }
  
  # Interpolation (Requirement 3)
  # We fill gaps: if t-1 and t+1 exist, but t is empty/null, average them.
  for (i in 2:(length(raw_mats) - 1)) {
    if (!has_data[i] && has_data[i-1] && has_data[i+1]) {
      # Linear interpolation
      raw_mats[[i]] <- (raw_mats[[i-1]] + raw_mats[[i+1]]) / 2
      has_data[i] <- TRUE
      # msg("    -> Interpolated missing frame at index %d", indices[i])
    }
  }
  
  # Final check: Ensure dimensions are consistent
  valid_mats <- raw_mats[has_data]
  if (length(valid_mats) == 0) return(list(valid=FALSE, reason="No valid data after cleanup"))
  
  nr <- nrow(valid_mats[[1]])
  nc <- ncol(valid_mats[[1]])
  
  list(valid = TRUE, mats = raw_mats, indices = indices, nr = nr, nc = nc, has_data = has_data)
}

# Connected Components (Preserved)
label_components_4n <- function(mask) {
  nr <- nrow(mask); nc <- ncol(mask)
  if (nr == 0 || nc == 0) return(list())
  lab <- matrix(0L, nr, nc)
  comps <- list(); lab_id <- 0L
  nbors <- matrix(c(-1,0, 1,0, 0,-1, 0,1), ncol = 2, byrow = TRUE)
  
  for (r in 1:nr) {
    for (c in 1:nc) {
      if (!mask[r, c] || lab[r, c] != 0L) next
      lab_id <- lab_id + 1L
      qr <- integer(nr*nc); qc <- integer(nr*nc); head <- 1L; tail <- 1L
      qr[tail] <- r; qc[tail] <- c; tail <- tail + 1L
      lab[r, c] <- lab_id
      members <- integer(0)
      while (head < tail) {
        rr <- qr[head]; cc <- qc[head]; head <- head + 1L
        members <- c(members, (cc - 1L) * nr + rr)
        for (k in 1:4) {
          r2 <- rr + nbors[k, 1]; c2 <- cc + nbors[k, 2]
          if (r2 >= 1 && r2 <= nr && c2 >= 1 && c2 <= nc && mask[r2, c2] && lab[r2, c2] == 0L) {
            lab[r2, c2] <- lab_id
            qr[tail] <- r2; qc[tail] <- c2; tail <- tail + 1L
          }
        }
      }
      comps[[lab_id]] <- members
    }
  }
  comps
}

# Helper: Get Weighted Centroid (Row, Col) of a component
get_centroid <- function(comp_indices, mat, nr) {
  if (length(comp_indices) == 0) return(c(NA, NA))
  
  rows <- ((comp_indices - 1L) %% nr) + 1L
  cols <- ((comp_indices - 1L) %/% nr) + 1L
  
  vals <- mat[comp_indices]
  total_mass <- sum(vals)
  
  if (total_mass == 0) return(c(mean(rows), mean(cols)))
  
  w_r <- sum(rows * vals) / total_mass
  w_c <- sum(cols * vals) / total_mass
  c(w_r, w_c)
}

# Helper: IoU
iou_sets <- function(a, b) {
  if (length(a) == 0 || length(b) == 0) return(0)
  length(intersect(a, b)) / length(unique(c(a, b)))
}

# ---------------------- Graph Construction ----------------------

build_networks_for_intervals <- function(cfg = CFG) {
  # Resolve Timestamp
  ts_path <- tryCatch(
    file.path(cfg$base_dir, ".data", cfg$participant, sprintf("%s_timestamp_merge.txt", cfg$gesture)),
    error = function(e) stop("Path error")
  )
  if (!file.exists(ts_path)) {
    # Try alternative path
    ts_path <- file.path(cfg$base_dir, "data", cfg$participant, sprintf("%s_timestamp_merge.txt", cfg$gesture))
    print(ts_path)
    if(!file.exists(ts_path)) stop("Timestamp file not found.")
  }
  
  # Parse Intervals
  lines <- readLines(ts_path)
  intervals <- list()
  for (ln in lines) {
    nums <- as.integer(regmatches(ln, gregexpr("\\d+", ln))[[1]])
    if (length(nums) >= 2) intervals[[length(intervals)+1]] <- c(nums[1], nums[2])
  }
  
  # Output Setup
  dir.create(file.path(cfg$out_dir, "png"), recursive = TRUE, showWarnings = FALSE)
  dir.create(file.path(cfg$out_dir, "csv"), recursive = TRUE, showWarnings = FALSE)
  
  # --- Process Each Interval ---
  for (k in seq_along(intervals)) {
    iv <- intervals[[k]]
    
    # 1. Load & Clean Data (Filtering & Interpolation)
    seq_data <- load_and_clean_sequence(cfg, iv[1], iv[2])
    
    if (!seq_data$valid) {
      msg("Skipping Interval %d (%d-%d): %s", k, iv[1], iv[2], seq_data$reason)
      next
    }
    
    msg("Processing Interval %d (%d-%d) | Frames: %d", k, iv[1], iv[2], length(seq_data$indices))
    
    mats <- seq_data$mats
    nr <- seq_data$nr
    nc <- seq_data$nc
    all_edges <- list()
    
    # 2. Traverse Frames
    for (i in 1:(length(mats) - 1)) {
      if (!seq_data$has_data[i] || !seq_data$has_data[i+1]) next
      
      mat_t  <- mats[[i]]
      mat_t1 <- mats[[i+1]]
      
      comps_t  <- label_components_4n(mat_t > 0)
      comps_t1 <- label_components_4n(mat_t1 > 0)
      
      # Match Blobs via IoU
      # Simple greedy match
      used_j <- rep(FALSE, length(comps_t1))
      
      for (idx_a in seq_along(comps_t)) {
        best_iou <- -1
        best_idx_b <- NA
        
        for (idx_b in seq_along(comps_t1)) {
          if (used_j[idx_b]) next
          val <- iou_sets(comps_t[[idx_a]], comps_t1[[idx_b]])
          if (val > best_iou) { best_iou <- val; best_idx_b <- idx_b }
        }
        
        if (!is.na(best_idx_b) && best_iou > cfg$iou_threshold) {
          used_j[best_idx_b] <- TRUE
          
          # --- NEW LOGIC: Displacement Vector (Point 4 & 1) ---
          
          # Calculate Centroids
          cent_t  <- get_centroid(comps_t[[idx_a]], mat_t, nr)
          cent_t1 <- get_centroid(comps_t1[[best_idx_b]], mat_t1, nr)
          
          # Vector (dy, dx)
          vec <- cent_t1 - cent_t
          magnitude <- sqrt(sum(vec^2))
          
          # Only draw edges if movement is significant (Point 1)
          if (magnitude > cfg$movement_threshold) {
            
            comp_indices_t <- comps_t[[idx_a]]
            
            # Round vector to nearest integer for grid mapping
            shift_r <- round(vec[1])
            shift_c <- round(vec[2])
            
            # Map every active cell in T to its projected position in T+1
            for (lin_idx in comp_indices_t) {
              r <- ((lin_idx - 1L) %% nr) + 1L
              c <- ((lin_idx - 1L) %/% nr) + 1L
              
              target_r <- r + shift_r
              target_c <- c + shift_c
              
              # Check bounds
              if (target_r >= 1 && target_r <= nr && target_c >= 1 && target_c <= nc) {
                target_lin <- (target_c - 1L) * nr + target_r
                
                # Optional: Strictly require target to be active in T+1?
                # Let's require target to be non-zero to keep graph clean
                if (mat_t1[target_r, target_c] > 0) {
                  all_edges[[length(all_edges)+1]] <- c(lin_idx, target_lin)
                }
              }
            }
          } # end if moving
        }
      }
    }
    
    # 3. Build Graph
    if (length(all_edges) > 0) {
      edges_mat <- do.call(rbind, all_edges)
    } else {
      edges_mat <- matrix(character(0), ncol=2)
    }
    
    # Grid Layout Nodes
    all_ids <- 1:(nr*nc)
    rc <- arrayInd(all_ids, .dim = c(nr, nc))
    V_df <- data.frame(name=as.character(all_ids), x=rc[,2], y=nr-rc[,1]+1) # y flipped
    
    # Create Graph
    g <- graph_from_data_frame(as.data.frame(edges_mat, stringsAsFactors=FALSE), vertices=V_df, directed=TRUE)
    
    # Save CSV
    csv_name <- file.path(cfg$out_dir, "csv", sprintf("%s_%s_int%d.csv", cfg$participant, cfg$gesture, k))
    if(nrow(edges_mat) > 0) utils::write.csv(edges_mat, csv_name, row.names=FALSE)
    
    # Plot
    if (cfg$plot_png) {
      png_name <- file.path(cfg$out_dir, "png", sprintf("%s_%s_int%d.png", cfg$participant, cfg$gesture, k))
      png(png_name, width=cfg$png_width, height=cfg$png_height)
      
      # Only plot active nodes/edges to keep it clean
      deg <- degree(g, mode="all")
      keep_v <- V(g)[deg > 0]
      
      plot(g, 
           vertex.size=3, 
           vertex.label=NA, 
           vertex.color=ifelse(deg>0, "black", NA), 
           vertex.frame.color=NA,
           edge.arrow.size=0.4,
           edge.color="red",
           main=sprintf("Flow Graph: Interval %d (Shift > %.1f)", k, cfg$movement_threshold))
      dev.off()
    }
  }
}

# Run
build_networks_for_intervals(CFG)