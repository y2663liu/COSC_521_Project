library(igraph)

# ---------------------- Configuration ----------------------
CFG <- list(
  base_dir           = "./../PressureSensorPi/",
  
  # --- Pre-processing Params ---
  use_pooling        = FALSE,          # TRUE = 2x2 Max Pool (32x32), FALSE = Raw (64x64)
  min_pressure_sum   = 100,            # Filter: Frame valid only if sum of pixels > this
  min_seq_len        = 10,             # Filter: Min frames for valid sequence
  
  # --- Network / Flow Params ---
  iou_threshold      = 0.20,           # 1. Primary Match: Min Overlap to match blobs
  max_match_dist     = 5.0,           # 2. Fallback Match: Max Distance (pixels) if IoU is 0
  movement_threshold = 0.5,           # Min pixel shift to draw edge (Filters stationary)
  edge_mode          = "flow",         # "flow" = centroid vector
  
  # --- Output Params ---
  out_dir            = "networks_flow",
  plot_png           = TRUE,
  png_width          = 1200,
  png_height         = 900,
  png_pointsize      = 14
)

# ---------------------- 1. Discovery Utilities ----------------------
msg <- function(...) cat(sprintf(...), "\n")

list_participants <- function(base_dir) {
  p1 <- file.path(base_dir, ".data")
  p2 <- file.path(base_dir, "data")
  d1 <- if (dir.exists(p1)) list.dirs(p1, full.names = FALSE, recursive = FALSE) else character()
  d2 <- if (dir.exists(p2)) list.dirs(p2, full.names = FALSE, recursive = FALSE) else character()
  sort(unique(c(d1, d2)))
}

list_gestures <- function(base_dir, participant) {
  roots <- c(file.path(base_dir, ".data", participant),
             file.path(base_dir, "data",  participant))
  roots <- roots[dir.exists(roots)]
  if (!length(roots)) return(character())
  
  files <- unlist(lapply(roots, function(r)
    list.files(r, pattern = "_timestamp(_merge)?\\.txt$", full.names = FALSE)
  ))
  if (!length(files)) return(character())
  
  gestures <- unique(sub("_timestamp(_merge)?\\.txt$", "", files))
  
  keep <- vapply(gestures, function(g) {
    hp <- file.path(base_dir, "harsh_process", participant, g)
    if (!dir.exists(hp)) return(FALSE)
    any(grepl("^f\\d{4,5}\\.csv$", list.files(hp)[1:10])) 
  }, logical(1))
  
  sort(gestures[keep])
}

resolve_timestamp_file <- function(base_dir, participant, gesture) {
  candidates <- c(
    file.path(base_dir, ".data", participant, sprintf("%s_timestamp_merge.txt", gesture)),
    file.path(base_dir, "data",  participant, sprintf("%s_timestamp_merge.txt", gesture)),
    file.path(base_dir, ".data", participant, sprintf("%s_timestamp.txt",        gesture)),
    file.path(base_dir, "data",  participant, sprintf("%s_timestamp.txt",        gesture))
  )
  existing <- candidates[file.exists(candidates)]
  if (length(existing) == 0) stop("No timestamp file found")
  existing[[1]]
}

parse_intervals <- function(ts_path) {
  lines <- readLines(ts_path, warn = FALSE)
  intervals <- list()
  for (ln in lines) {
    if (startsWith(trimws(ln), "#") || !nzchar(trimws(ln))) next
    nums <- as.integer(regmatches(ln, gregexpr("\\d+", ln))[[1]])
    if (length(nums) >= 2) intervals[[length(intervals)+1]] <- c(min(nums), max(nums))
  }
  intervals
}

# ---------------------- 2. Image Processing Utilities ----------------------
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

read_frame_matrix <- function(cfg, frame_idx) {
  fn <- file.path(cfg$base_dir, "harsh_process", cfg$participant, cfg$gesture, sprintf("f%05d.csv", frame_idx))
  if (!file.exists(fn)) return(NULL)
  
  tryCatch({
    mat <- as.matrix(utils::read.csv(fn, header = FALSE, check.names = FALSE))
    if (cfg$use_pooling) return(pool_2x2(mat)) else return(mat)
  }, error = function(e) NULL)
}

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

# ---------------------- 3. Advanced Logic ----------------------

iou_sets <- function(a, b) {
  if (length(a) == 0 || length(b) == 0) return(0)
  length(intersect(a, b)) / length(unique(c(a, b)))
}

get_centroid <- function(comp_indices, mat, nr) {
  if (length(comp_indices) == 0) return(c(NA, NA))
  rows <- ((comp_indices - 1L) %% nr) + 1L
  cols <- ((comp_indices - 1L) %/% nr) + 1L
  vals <- mat[comp_indices]
  total_mass <- sum(vals)
  if (total_mass == 0) return(c(mean(rows), mean(cols)))
  c(sum(rows * vals) / total_mass, sum(cols * vals) / total_mass)
}

load_and_clean_sequence <- function(cfg, start_f, end_f) {
  indices <- start_f:end_f
  if (length(indices) < cfg$min_seq_len) return(list(valid = FALSE, reason = "Sequence too short"))
  
  raw_mats <- vector("list", length(indices))
  has_data <- logical(length(indices))
  
  # 1. Load Raw Data
  for (i in seq_along(indices)) {
    m <- read_frame_matrix(cfg, indices[i])
    if (!is.null(m)) {
      raw_mats[[i]] <- m
      if (sum(m, na.rm=TRUE) > cfg$min_pressure_sum) has_data[i] <- TRUE
    }
  }
  
  # Check if empty before interpolation
  if (!any(has_data)) return(list(valid = FALSE, reason = "No valid frames"))
  
  # 2. Multi-Frame Interpolation
  valid_indices <- which(has_data)
  
  # We need at least two valid frames to interpolate between them
  if (length(valid_indices) >= 2) {
    for (k in 1:(length(valid_indices) - 1)) {
      idx1 <- valid_indices[k]
      idx2 <- valid_indices[k+1]
      
      # If there is a gap greater than 0 (indices are not adjacent)
      if (idx2 > idx1 + 1) {
        mat_start <- raw_mats[[idx1]]
        mat_end   <- raw_mats[[idx2]]
        gap_len   <- idx2 - idx1
        
        # Fill the gap
        for (j in 1:(gap_len - 1)) {
          target_idx <- idx1 + j
          
          # Linear Interpolation Formula:
          # val = start * (1 - alpha) + end * alpha
          alpha <- j / gap_len 
          
          raw_mats[[target_idx]] <- (1 - alpha) * mat_start + alpha * mat_end
          has_data[target_idx]   <- TRUE
        }
      }
    }
  }
  
  # 3. Final Output Selection
  valid_mats <- raw_mats[has_data]
  
  if (length(valid_mats) == 0) return(list(valid=FALSE, reason="No valid data"))
  
  list(valid = TRUE, mats = raw_mats, indices = indices, 
       nr = nrow(valid_mats[[1]]), nc = ncol(valid_mats[[1]]), has_data = has_data)
}

# ---------------------- 4. Core Network Builder ----------------------
build_networks_for_intervals <- function(cfg) {
  ts_path <- resolve_timestamp_file(cfg$base_dir, cfg$participant, cfg$gesture)
  intervals <- parse_intervals(ts_path)
  if (length(intervals) == 0) stop("No intervals found")
  
  pool_lbl <- if(cfg$use_pooling) "pool" else "raw"
  subdir_name <- sprintf("%s_iou%.2f_move%.2f_dist%.0f", pool_lbl, cfg$iou_threshold, cfg$movement_threshold, cfg$max_match_dist)
  
  out_png_dir <- file.path(cfg$out_dir, subdir_name, cfg$participant, cfg$gesture, "png")
  out_csv_dir <- file.path(cfg$out_dir, subdir_name, cfg$participant, cfg$gesture, "csv")
  dir.create(out_png_dir, recursive = TRUE, showWarnings = FALSE)
  dir.create(out_csv_dir, recursive = TRUE, showWarnings = FALSE)
  
  for (k in seq_along(intervals)) {
    iv <- intervals[[k]]
    seq_data <- load_and_clean_sequence(cfg, iv[1], iv[2])
    if (!seq_data$valid) next
    
    mats <- seq_data$mats
    nr <- seq_data$nr
    nc <- seq_data$nc
    all_edges <- list()
    all_active_indices <- integer(0)
    
    for (i in 1:(length(mats) - 1)) {
      if (!seq_data$has_data[i] || !seq_data$has_data[i+1]) next
      
      mat_t  <- mats[[i]]
      mat_t1 <- mats[[i+1]]
      comps_t  <- label_components_4n(mat_t > 0)
      comps_t1 <- label_components_4n(mat_t1 > 0)
      
      if(length(comps_t) > 0) all_active_indices <- c(all_active_indices, unlist(comps_t))
      
      # Pre-calculate Centroids for all blobs in both frames
      cents_t  <- lapply(comps_t,  function(c) get_centroid(c, mat_t, nr))
      cents_t1 <- lapply(comps_t1, function(c) get_centroid(c, mat_t1, nr))
      
      # MATCHING LOGIC
      used_j <- rep(FALSE, length(comps_t1))
      
      for (idx_a in seq_along(comps_t)) {
        matched_idx_b <- NA
        
        # 1. Primary Match: IoU
        best_iou <- -1
        cand_iou <- NA
        for (idx_b in seq_along(comps_t1)) {
          if (used_j[idx_b]) next
          val <- iou_sets(comps_t[[idx_a]], comps_t1[[idx_b]])
          if (val > best_iou) { best_iou <- val; cand_iou <- idx_b }
        }
        
        if (!is.na(cand_iou) && best_iou > cfg$iou_threshold) {
          matched_idx_b <- cand_iou
        } else {
          # 2. Fallback Match: Nearest Distance (if IoU failed)
          best_dist <- Inf
          cand_dist <- NA
          c_a <- cents_t[[idx_a]]
          
          for (idx_b in seq_along(comps_t1)) {
            if (used_j[idx_b]) next # Don't steal already matched blobs
            
            c_b <- cents_t1[[idx_b]]
            dist <- sqrt(sum((c_b - c_a)^2))
            if (dist < best_dist) { best_dist <- dist; cand_dist <- idx_b }
          }
          
          if (!is.na(cand_dist) && best_dist < cfg$max_match_dist) {
            matched_idx_b <- cand_dist
          }
        }
        
        # PROCEED IF MATCH FOUND
        if (!is.na(matched_idx_b)) {
          used_j[matched_idx_b] <- TRUE
          
          c_a <- cents_t[[idx_a]]
          c_b <- cents_t1[[matched_idx_b]]
          vec <- c_b - c_a
          magnitude <- sqrt(sum(vec^2))
          
          if (magnitude > cfg$movement_threshold) {
            shift_r <- round(vec[1])
            shift_c <- round(vec[2])
            
            for (lin_idx in comps_t[[idx_a]]) {
              r <- ((lin_idx - 1L) %% nr) + 1L
              c <- ((lin_idx - 1L) %/% nr) + 1L
              tr <- r + shift_r; tc <- c + shift_c
              
              if (tr >= 1 && tr <= nr && tc >= 1 && tc <= nc) {
                if (mat_t1[tr, tc] > 0) {
                  t_lin <- (tc - 1L) * nr + tr
                  all_edges[[length(all_edges)+1]] <- c(lin_idx, t_lin)
                }
              }
            }
          }
        }
      }
    }
    
    last_mat <- mats[[length(mats)]]
    last_comps <- label_components_4n(last_mat > 0)
    if(length(last_comps) > 0) all_active_indices <- c(all_active_indices, unlist(last_comps))
    all_active_indices <- unique(all_active_indices)
    
    if (length(all_edges) > 0) edges_mat <- do.call(rbind, all_edges) else edges_mat <- matrix(character(0), ncol=2)
    
    all_ids <- 1:(nr*nc)
    rc <- arrayInd(all_ids, .dim = c(nr, nc))
    V_df <- data.frame(name=as.character(all_ids), x=rc[,2], y=nr-rc[,1]+1)
    
    g <- graph_from_data_frame(as.data.frame(edges_mat, stringsAsFactors=FALSE), vertices=V_df, directed=TRUE)
    
    csv_name <- file.path(out_csv_dir, sprintf("int_%03d.csv", k))
    if(nrow(edges_mat) > 0) utils::write.csv(edges_mat, csv_name, row.names=FALSE)
    
    if (cfg$plot_png) {
      png_name <- file.path(out_png_dir, sprintf("int_%03d.png", k))
      png(png_name, width=cfg$png_width, height=cfg$png_height, pointsize=cfg$png_pointsize)
      
      is_active <- V(g)$name %in% as.character(all_active_indices)
      v_cols <- ifelse(is_active, "black", NA)
      
      plot(g, layout=cbind(V(g)$x, V(g)$y),
           vertex.size=3, vertex.label=NA, vertex.color=v_cols, vertex.frame.color=NA,
           edge.arrow.size=0.4, edge.color="red",
           main=sprintf("%s/%s Int %d (Active:%d)", cfg$participant, cfg$gesture, k, length(all_active_indices)))
      dev.off()
    }
  }
}

# ---------------------- 5. Batch Drivers ----------------------

run_one_pair <- function(base_cfg, participant, gesture) {
  run_cfg <- base_cfg
  run_cfg$participant <- participant
  run_cfg$gesture     <- gesture
  msg("=== Processing: %s / %s ===", participant, gesture)
  tryCatch({
    build_networks_for_intervals(run_cfg)
    TRUE
  }, error = function(e) {
    msg("!! Error %s / %s: %s", participant, gesture, conditionMessage(e))
    FALSE
  })
}

build_all_networks <- function(cfg = CFG) {
  participants <- list_participants(cfg$base_dir)
  if (!length(participants)) stop("No participants found.")
  
  total <- 0L; ok <- 0L
  for (p in participants) {
    gestures <- list_gestures(cfg$base_dir, p)
    if (!length(gestures)) next
    for (g in gestures) {
      total <- total + 1L
      ok <- ok + as.integer(run_one_pair(cfg, p, g))
    }
  }
  msg("=== Batch Complete: %d/%d successful. ===", ok, total)
}

build_all_networks(CFG)