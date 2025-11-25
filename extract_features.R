library(igraph)
library(e1071) # For skewness/kurtosis
library(dplyr)
library(grDevices) # For chull

# ---------------------- Configuration ----------------------
CFG <- list(
  base_dir        = "./../PressureSensorPi/",
  network_dir     = "networks",
  input_subdir    = "raw_iou0.20_move0.50_dist5_simple", 
  output_dir      = "train_data"
)

msg <- function(...) cat(sprintf(...), "\n")

# ---------------------- Utilities ----------------------

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

poly_area <- function(x, y) {
  if (length(x) < 3) return(0)
  0.5 * abs(sum(x * c(y[-1], y[1]) - y * c(x[-1], x[1])))
}

# ---------------------- 1. Raw Spatio-Temporal Stats ----------------------
get_raw_stats <- function(base_dir, participant, gesture, interval_idx) {
  ts_paths <- c(
    file.path(base_dir, ".data", participant, sprintf("%s_timestamp_merge.txt", gesture)),
    file.path(base_dir, "data",  participant, sprintf("%s_timestamp_merge.txt", gesture)),
    file.path(base_dir, ".data", participant, sprintf("%s_timestamp.txt",        gesture)),
    file.path(base_dir, "data",  participant, sprintf("%s_timestamp.txt",        gesture))
  )
  ts_path <- ts_paths[file.exists(ts_paths)][1]
  if (is.na(ts_path)) return(NULL) 
  
  lines <- readLines(ts_path, warn=FALSE)
  valid_lines <- lines[!startsWith(trimws(lines), "#") & nzchar(trimws(lines))]
  
  if (interval_idx > length(valid_lines)) return(NULL)
  nums <- as.integer(regmatches(valid_lines[interval_idx], gregexpr("\\d+", valid_lines[interval_idx]))[[1]])
  if (length(nums) < 2) return(NULL)
  
  start_f <- min(nums); end_f <- max(nums)
  duration <- end_f - start_f + 1
  
  max_vals <- numeric(0)
  mean_vals <- numeric(0)
  median_vals <- numeric(0)
  blob_counts <- numeric(0)
  
  seen_nodes <- integer(0)
  node_activity_counts <- list() 
  all_active_coords <- list()
  
  nr <- 0; nc <- 0
  
  for (f in start_f:end_f) {
    fn <- file.path(base_dir, "harsh_process", participant, gesture, sprintf("f%05d.csv", f))
    if (file.exists(fn)) {
      mat <- tryCatch(as.matrix(utils::read.csv(fn, header=FALSE)), error=function(e) NULL)
      if (!is.null(mat)) {
        if(nr == 0) { nr <- nrow(mat); nc <- ncol(mat) }
        
        active_mask <- mat > 0
        active_vals <- mat[active_mask]
        
        if (length(active_vals) > 0) {
          max_vals <- c(max_vals, max(active_vals))
          mean_vals <- c(mean_vals, mean(active_vals))
          median_vals <- c(median_vals, median(active_vals))
          
          comps <- label_components_4n(active_mask)
          blob_counts <- c(blob_counts, length(comps))
          
          active_indices <- which(active_mask)
          
          for(idx in active_indices) {
            idx_char <- as.character(idx)
            node_activity_counts[[idx_char]] <- (node_activity_counts[[idx_char]] %||% 0) + 1
          }
          
          new_nodes <- setdiff(active_indices, seen_nodes)
          seen_nodes <- unique(c(seen_nodes, active_indices))
          
          rows <- (active_indices - 1) %% nr + 1
          cols <- (active_indices - 1) %/% nr + 1
          all_active_coords[[length(all_active_coords)+1]] <- cbind(rows, cols)
        } else {
          blob_counts <- c(blob_counts, 0)
        }
      }
    }
  }
  
  avg_persistence <- if(length(node_activity_counts) > 0) mean(unlist(node_activity_counts)) else 0
  activation_rate <- if(duration > 0) length(seen_nodes) / duration else 0
  
  hull_area <- 0
  max_phys_dist <- 0
  
  if (length(all_active_coords) > 0) {
    all_pts <- do.call(rbind, all_active_coords)
    all_pts <- unique(all_pts) 
    
    if (nrow(all_pts) >= 3) {
      hull_idx <- chull(all_pts[,1], all_pts[,2])
      hull_pts <- all_pts[hull_idx, ]
      hull_area <- poly_area(hull_pts[,1], hull_pts[,2])
    }
    
    if (nrow(all_pts) >= 2) {
      dists <- dist(all_pts)
      max_phys_dist <- max(dists)
    }
  }
  
  c(
    duration = duration,
    max_pressure = if(length(max_vals)>0) max(max_vals) else 0,
    avg_pressure = if(length(mean_vals)>0) mean(mean_vals) else 0,
    median_pressure = if(length(median_vals)>0) median(median_vals) else 0,
    avg_blobs    = if(length(blob_counts)>0) mean(blob_counts) else 0,
    max_blobs    = if(length(blob_counts)>0) max(blob_counts) else 0,
    persistence  = avg_persistence,
    activation_rate = activation_rate,
    contact_area = hull_area,
    max_phys_dist = max_phys_dist
  )
}

`%||%` <- function(a, b) if (is.null(a)) b else a

# ---------------------- 2. Advanced Graph Metrics ----------------------
compute_graph_metrics <- function(edges_path, nodes_path) {
  
  edges <- tryCatch(read.csv(edges_path, stringsAsFactors=FALSE), error=function(e) NULL)
  nodes <- tryCatch(read.csv(nodes_path, stringsAsFactors=FALSE), error=function(e) NULL)
  
  if (is.null(nodes) || nrow(nodes) == 0) {
    g <- make_empty_graph(directed=TRUE)
  } else {
    if (is.null(edges) || nrow(edges) == 0) edges <- data.frame(from=character(), to=character())
    nodes$name <- as.character(nodes$name)
    edges$from <- as.character(edges$from)
    edges$to   <- as.character(edges$to)
    g <- graph_from_data_frame(d = edges, vertices = nodes, directed = TRUE)
  }
  
  n_nodes <- vcount(g)
  n_edges <- ecount(g)
  
  # Default Values
  res <- list(
    n_nodes = n_nodes, n_edges = n_edges, density = 0, reciprocity = 0,
    n_comps = 0, max_comp_sz = 0, scc_count = 0,
    mean_deg = 0, sd_deg = 0, kurt_deg = 0, max_deg_in = 0, max_deg_out = 0, centralization = 0,
    mean_betweenness = 0, max_betweenness = 0,
    mean_closeness = 0, max_closeness = 0,
    max_hub = 0, max_auth = 0,
    avg_path_len = 0, median_path_len = 0, diameter = 0, clustering = 0
  )
  
  if (n_nodes > 0) {
    res$density <- edge_density(g)
    rec <- reciprocity(g)
    res$reciprocity <- if(is.na(rec)) 0 else rec
    
    comps <- components(g, mode="weak")
    res$n_comps <- comps$no
    res$max_comp_sz <- max(comps$csize)
    
    scc <- components(g, mode="strong")
    res$scc_count <- scc$no
    
    deg_all <- degree(g, mode="all")
    deg_in  <- degree(g, mode="in")
    deg_out <- degree(g, mode="out")
    
    res$mean_deg <- mean(deg_all)
    res$sd_deg   <- sd(deg_all)
    kurt <- kurtosis(deg_all, na.rm=TRUE)
    res$kurt_deg <- if(is.nan(kurt)) 0 else kurt
    res$max_deg_in <- max(deg_in)
    res$max_deg_out <- max(deg_out)
    
    centr <- centr_degree(g, mode="all")$centralization
    res$centralization <- if(is.na(centr)) 0 else centr
    
    bw <- betweenness(g, normalized = TRUE)
    res$mean_betweenness <- mean(bw)
    res$max_betweenness  <- max(bw)
    
    cl <- closeness(g, normalized = TRUE)
    cl[is.nan(cl)] <- 0
    res$mean_closeness <- mean(cl)
    res$max_closeness  <- max(cl)
    
    try({
      hs <- hub_score(g)$vector
      as <- authority_score(g)$vector
      res$max_hub <- if(length(hs)>0) max(hs) else 0
      res$max_auth <- if(length(as)>0) max(as) else 0
    }, silent=TRUE)
    
    # --- PATH METRICS ---
    if (n_edges > 0) {
      # 1. Extract Giant Component (to avoid Infinite distances from disconnected parts)
      giant_g <- induced_subgraph(g, which(comps$membership == which.max(comps$csize)))
      
      # 2. Get Full Distance Matrix for Giant Component
      # mode="out" follows direction of flow
      d_mat <- distances(giant_g, mode="out")
      
      # 3. Filter valid paths (remove Inf and diagonal 0s)
      valid_dists <- d_mat[is.finite(d_mat) & d_mat > 0]
      
      if(length(valid_dists) > 0) {
        res$avg_path_len    <- mean(valid_dists)
        res$median_path_len <- median(valid_dists)
        res$diameter        <- max(valid_dists)
      }
      
      tr <- transitivity(giant_g, type="global")
      res$clustering <- if(is.na(tr)) 0 else tr
    }
  }
  
  return(res)
}

# ---------------------- Main Loop ----------------------
process_all <- function() {
  input_dir <- file.path(CFG$network_dir, CFG$input_subdir)
  edge_files <- list.files(input_dir, pattern = "_edges\\.csv$", recursive = TRUE, full.names = TRUE)
  
  if (length(edge_files) == 0) stop("No _edges.csv files found in ", input_dir)
  
  results <- list()
  msg("Found %d edge files. Extracting extended features...", length(edge_files))
  
  count <- 0
  for (edge_path in edge_files) {
    count <- count + 1
    if (count %% 20 == 0) msg("Processed %d / %d", count, length(edge_files))
    
    node_path <- sub("_edges\\.csv$", "_nodes.csv", edge_path)
    parts <- strsplit(edge_path, "/")[[1]]
    fname <- parts[length(parts)]
    gesture <- parts[length(parts) - 2]
    participant <- parts[length(parts) - 3]
    
    interval_idx <- as.integer(sub("_edges.csv", "", sub("int_", "", fname)))
    if(is.na(interval_idx)) interval_idx <- 1
    
    g_feats <- compute_graph_metrics(edge_path, node_path)
    
    raw_feats <- get_raw_stats(CFG$base_dir, participant, gesture, interval_idx)
    if(is.null(raw_feats)) {
      raw_feats <- list(duration=0, max_pressure=0, avg_pressure=0, median_pressure=0, avg_blobs=0, max_blobs=0, 
                        persistence=0, activation_rate=0, contact_area=0, max_phys_dist=0)
    }
    
    results[[length(results)+1]] <- c(
      list(participant = participant, gesture = gesture),
      as.list(raw_feats),
      g_feats
    )
  }
  
  final_df <- bind_rows(results)
  final_df[is.na(final_df)] <- 0
  
  dir.create(CFG$output_dir, recursive = TRUE, showWarnings = FALSE)
  out_file <- sprintf("%s/%s_features_extended.csv", CFG$output_dir, basename(CFG$input_subdir))
  write.csv(final_df, out_file, row.names = FALSE)
  msg("Done! Extended features saved to %s", out_file)
}

process_all()