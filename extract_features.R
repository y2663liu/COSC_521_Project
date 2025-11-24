library(igraph)
library(e1071)
library(dplyr)

# ---------------------- Configuration ----------------------
CFG <- list(
  base_dir        = "./../PressureSensorPi/",
  network_dir     = "networks",
  input_subdir    = "raw_iou0.20_move0.50_dist5_simple",
  output_dir      = "train_data"
)

msg <- function(...) cat(sprintf(...), "\n")

# Helper: Raw Stats
get_raw_stats <- function(base_dir, participant, gesture, interval_idx) {
  ts_paths <- c(
    file.path(base_dir, ".data", participant, sprintf("%s_timestamp_merge.txt", gesture)),
    file.path(base_dir, "data",  participant, sprintf("%s_timestamp_merge.txt", gesture)),
    file.path(base_dir, ".data", participant, sprintf("%s_timestamp.txt",        gesture)),
    file.path(base_dir, "data",  participant, sprintf("%s_timestamp.txt",        gesture))
  )
  ts_path <- ts_paths[file.exists(ts_paths)][1]
  if (is.na(ts_path)) return(c(dur=NA, max_p=NA, mean_p=NA))
  
  lines <- readLines(ts_path, warn=FALSE)
  valid_lines <- lines[!startsWith(trimws(lines), "#") & nzchar(trimws(lines))]
  
  if (interval_idx > length(valid_lines)) return(c(dur=NA, max_p=NA, mean_p=NA))
  nums <- as.integer(regmatches(valid_lines[interval_idx], gregexpr("\\d+", valid_lines[interval_idx]))[[1]])
  if (length(nums) < 2) return(c(dur=NA, max_p=NA, mean_p=NA))
  
  start_f <- min(nums); end_f <- max(nums)
  duration <- end_f - start_f + 1
  
  max_vals <- numeric(0); mean_vals <- numeric(0); median_vals <- numeric(0);
  for (f in start_f:end_f) {
    fn <- file.path(base_dir, "harsh_process", participant, gesture, sprintf("f%05d.csv", f))
    if (file.exists(fn)) {
      mat <- tryCatch(as.matrix(utils::read.csv(fn, header=FALSE)), error=function(e) NULL)
      if (!is.null(mat)) {
        max_vals <- c(max_vals, max(mat, na.rm=TRUE))
        mean_vals <- c(mean_vals, mean(mat[mat>0], na.rm=TRUE)) 
        median_vals <- c(median_vals, median(mat[mat>0], na.rm=TRUE)) 
      }
    }
  }
  c(duration = duration,
    max_pressure = if(length(max_vals)>0) max(max_vals) else 0,
    avg_pressure = if(length(mean_vals)>0) mean(mean_vals, na.rm=TRUE) else 0,
    median_pressure = if(length(median_vals)>0) median(median_vals, na.rm=TRUE) else 0
  )
}

# Helper: Graph Metrics
compute_graph_metrics <- function(edges_path, nodes_path) {
  
  # 1. Load Data
  edges <- tryCatch(read.csv(edges_path, stringsAsFactors=FALSE), error=function(e) NULL)
  nodes <- tryCatch(read.csv(nodes_path, stringsAsFactors=FALSE), error=function(e) NULL)
  
  # 2. Build Graph (Include isolated active nodes)
  if (is.null(nodes) || nrow(nodes) == 0) {
    # If no active nodes, graph is empty
    g <- make_empty_graph(directed=TRUE)
  } else {
    # If edges empty, ensure it's a valid empty DF
    if (is.null(edges) || nrow(edges) == 0) {
      edges <- data.frame(from=character(), to=character())
    }
    # Ensure columns match
    nodes$name <- as.character(nodes$name)
    edges$from <- as.character(edges$from)
    edges$to   <- as.character(edges$to)
    
    # Construct graph using Nodes explicitly
    g <- graph_from_data_frame(d = edges, vertices = nodes, directed = TRUE)
  }
  
  # 3. Metrics
  n_nodes <- vcount(g)
  n_edges <- ecount(g)
  density <- edge_density(g)
  reciprocity <- reciprocity(g) 
  if(is.na(reciprocity)) reciprocity <- 0
  
  # Components
  comps_weak <- components(g, mode="weak")
  n_comps    <- comps_weak$no
  max_comp_sz <- if(n_comps > 0) max(comps_weak$csize) else 0
  
  # Degrees
  if (n_nodes > 0) {
    deg_all <- degree(g, mode="all")
    deg_in  <- degree(g, mode="in")
    deg_out <- degree(g, mode="out")
    
    mean_deg <- mean(deg_all)
    max_deg  <- max(deg_all)
    skew_deg <- skewness(deg_all, na.rm=TRUE)
    
    n_sources <- sum(deg_out > 0 & deg_in == 0)
    n_sinks   <- sum(deg_in > 0 & deg_out == 0)
    ratio_source_sink <- (n_sources + n_sinks) / n_nodes
  } else {
    mean_deg <- 0; max_deg <- 0; skew_deg <- 0
    n_sources <- 0; n_sinks <- 0; ratio_source_sink <- 0
  }
  
  # Paths (on largest component)
  avg_path_len <- 0
  diam <- 0
  transitivity <- 0
  
  if (n_nodes > 1 && n_edges > 0) {
    giant_g <- induced_subgraph(g, which(comps_weak$membership == which.max(comps_weak$csize)))
    avg_path_len <- mean_distance(giant_g, directed=TRUE)
    diam         <- diameter(giant_g, directed=TRUE) 
    transitivity <- transitivity(giant_g, type="global")
    if(is.na(transitivity)) transitivity <- 0
  }
  
  list(
    n_nodes = n_nodes,
    n_edges = n_edges,
    density = density,
    reciprocity = reciprocity,
    n_comps = n_comps,
    max_comp_sz = max_comp_sz,
    mean_deg = mean_deg,
    max_deg = max_deg,
    skew_deg = ifelse(is.nan(skew_deg), 0, skew_deg),
    n_sources = n_sources,
    n_sinks = n_sinks,
    ratio_source_sink = ratio_source_sink,
    avg_path_len = avg_path_len,
    diameter = diam,
    clustering = transitivity
  )
}

process_all <- function() {
  input_dir <- file.path(CFG$network_dir, CFG$input_subdir)
  # Look for EDGES files
  edge_files <- list.files(input_dir, pattern = "_edges\\.csv$", recursive = TRUE, full.names = TRUE)
  
  if (length(edge_files) == 0) stop("No _edges.csv files found in ", input_dir)
  
  results <- list()
  msg("Found %d edge files. Extracting features...", length(edge_files))
  
  count <- 0
  for (edge_path in edge_files) {
    count <- count + 1
    if (count %% 50 == 0) msg("Processed %d / %d", count, length(edge_files))
    
    # Derive Node Path
    node_path <- sub("_edges\\.csv$", "_nodes.csv", edge_path)
    
    # Metadata
    parts <- strsplit(edge_path, "/")[[1]]
    fname <- parts[length(parts)]
    gesture <- parts[length(parts) - 2]
    participant <- parts[length(parts) - 3]
    
    # ID: int_001_edges.csv -> 1
    interval_idx <- as.integer(sub("_edges.csv", "", sub("int_", "", fname)))
    if(is.na(interval_idx)) interval_idx <- 1
    
    # 1. Compute
    g_feats <- compute_graph_metrics(edge_path, node_path)
    
    # 2. Raw Stats
    raw_feats <- get_raw_stats(CFG$base_dir, participant, gesture, interval_idx)
    
    # Combine
    results[[length(results)+1]] <- c(
      list(participant = participant, gesture = gesture),
      as.list(raw_feats),
      g_feats
    )
  }
  
  final_df <- bind_rows(results)
  final_df[is.na(final_df)] <- 0
  
  dir.create(CFG$output_dir, recursive = TRUE, showWarnings = FALSE)
  out_file <- sprintf("%s/%s.csv", CFG$output_dir, CFG$input_subdir)
  write.csv(final_df, out_file, row.names = FALSE)
  msg("Done! Features saved to %s", out_file)
}

process_all()