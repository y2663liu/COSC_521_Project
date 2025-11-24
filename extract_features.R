library(igraph)
library(e1071) # For skewness/kurtosis
library(dplyr)

# ---------------------- Configuration ----------------------
CFG <- list(
  base_dir        = "./../PressureSensorPi/",
  network_dir     = "networks",
  input_subdir    = "raw_iou0.20_move0.50_dist5_simple",
  output_dir      = "train_data"
)

# ---------------------- Helpers ----------------------
msg <- function(...) cat(sprintf(...), "\n")

# Helper to get raw pressure stats for the interval
get_raw_stats <- function(base_dir, participant, gesture, interval_idx) {
  # 1. Find timestamp file to get start/end frames
  ts_paths <- c(
    file.path(base_dir, ".data", participant, sprintf("%s_timestamp_merge.txt", gesture)),
    file.path(base_dir, "data",  participant, sprintf("%s_timestamp_merge.txt", gesture)),
    file.path(base_dir, ".data", participant, sprintf("%s_timestamp.txt",        gesture)),
    file.path(base_dir, "data",  participant, sprintf("%s_timestamp.txt",        gesture))
  )
  ts_path <- ts_paths[file.exists(ts_paths)][1]
  if (is.na(ts_path)) return(c(dur=NA, max_p=NA, mean_p=NA))
  
  # 2. Parse specific interval
  lines <- readLines(ts_path, warn=FALSE)
  valid_lines <- lines[!startsWith(trimws(lines), "#") & nzchar(trimws(lines))]
  
  if (interval_idx > length(valid_lines)) return(c(dur=NA, max_p=NA, mean_p=NA))
  
  nums <- as.integer(regmatches(valid_lines[interval_idx], gregexpr("\\d+", valid_lines[interval_idx]))[[1]])
  if (length(nums) < 2) return(c(dur=NA, max_p=NA, mean_p=NA))
  
  start_f <- min(nums); end_f <- max(nums)
  duration <- end_f - start_f + 1
  
  # 3. Scan frames for pressure stats
  max_vals <- numeric(0)
  mean_vals <- numeric(0)
  median_vals <- numeric(0)
  
  # We check a subset of frames to be fast, or all if feasible
  # Let's check all frames in interval
  for (f in start_f:end_f) {
    fn <- file.path(base_dir, "harsh_process", participant, gesture, sprintf("f%05d.csv", f))
    if (file.exists(fn)) {
      mat <- tryCatch(as.matrix(utils::read.csv(fn, header=FALSE)), error=function(e) NULL)
      if (!is.null(mat)) {
        max_vals <- c(max_vals, max(mat, na.rm=TRUE))
        mean_vals <- c(mean_vals, mean(mat[mat>0], na.rm=TRUE)) # Mean of active pixels only
        median_vals <- c(median_vals, median(mat[mat>0], na.rm=TRUE)) # Median of active pixels only
      }
    }
  }
  
  c(
    duration = duration,
    max_pressure = if(length(max_vals)>0) max(max_vals) else 0,
    avg_pressure = if(length(mean_vals)>0) mean(mean_vals, na.rm=TRUE) else 0,
    median_pressure = if(length(median_vals)>0) median(mean_vals, na.rm=TRUE) else 0
  )
}

# ---------------------- Feature Computation ----------------------
compute_graph_metrics <- function(csv_path) {
  # Load Edges
  edges <- tryCatch(read.csv(csv_path, stringsAsFactors=FALSE), error=function(e) NULL)
  
  # If empty graph
  if (is.null(edges) || nrow(edges) == 0) {
    g <- make_empty_graph(directed=TRUE)
  } else {
    g <- graph_from_data_frame(edges, directed=TRUE)
  }
  
  # --- Basic Topology ---
  n_nodes <- vcount(g)
  n_edges <- ecount(g)
  density <- edge_density(g)
  reciprocity <- reciprocity(g) # Crucial for Rub vs Swipe
  
  # --- Component Analysis (Crucial for 1 finger vs 3 fingers) ---
  comps_weak <- components(g, mode="weak")
  n_comps    <- comps_weak$no
  max_comp_sz <- if(n_comps > 0) max(comps_weak$csize) else 0
  
  # --- Degree Statistics ---
  if (n_nodes > 0) {
    deg_in  <- degree(g, mode="in")
    deg_out <- degree(g, mode="out")
    deg_all <- degree(g, mode="all")
    
    mean_deg <- mean(deg_all)
    max_deg  <- max(deg_all)
    sd_deg   <- sd(deg_all)
    skew_deg <- skewness(deg_all, na.rm=TRUE) # High for Pinch (converging flow)
    
    # Source/Sink Ratio (Flow Directionality)
    # Source: Out > 0, In = 0. Sink: In > 0, Out = 0.
    n_sources <- sum(deg_out > 0 & deg_in == 0)
    n_sinks   <- sum(deg_in > 0 & deg_out == 0)
    ratio_source_sink <- if(n_nodes>0) (n_sources + n_sinks) / n_nodes else 0
  } else {
    mean_deg <- 0; max_deg <- 0; sd_deg <- 0; skew_deg <- 0
    n_sources <- 0; n_sinks <- 0; ratio_source_sink <- 0
  }
  
  # --- Path / Distance Metrics ---
  # Warning: Shortest paths can be slow on massive graphs, but ours are sparse
  if (n_nodes > 1 && n_edges > 0) {
    # Only consider the largest component for meaningful path length
    giant_g <- induced_subgraph(g, which(components(g, mode="weak")$membership == which.max(components(g, mode="weak")$csize)))
    
    avg_path_len <- mean_distance(giant_g, directed=TRUE)
    diam         <- diameter(giant_g, directed=TRUE) # High for long Swipes
    transitivity <- transitivity(giant_g, type="global") # Clustering coefficient
    if(is.na(transitivity)) transitivity <- 0
  } else {
    avg_path_len <- 0
    diam <- 0
    transitivity <- 0
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

# ---------------------- Main Loop ----------------------
process_all <- function() {
  # Find all CSVs
  # Pattern: .../participant/gesture/csv/int_XXX.csv
  input_dir <- file.path(CFG$network_dir, CFG$input_subdir)
  csv_files <- list.files(input_dir, pattern = "\\.csv$", recursive = TRUE, full.names = TRUE)
  
  if (length(csv_files) == 0) stop("No CSV files found in ", CFG$input_subdir)
  
  results <- list()
  
  msg("Found %d network files. extracting features...", length(csv_files))
  
  count <- 0
  for (fpath in csv_files) {
    count <- count + 1
    if (count %% 50 == 0) msg("Processed %d / %d", count, length(csv_files))
    
    # Extract Metadata from path
    parts <- strsplit(fpath, "/")[[1]]
    # Assuming structure: ... / Participant / Gesture / csv / int_001.csv
    # We need to count backwards from the file name
    fname <- parts[length(parts)]
    gesture <- parts[length(parts) - 2]
    participant <- parts[length(parts) - 3]
    
    # Extract Interval ID from filename (int_001.csv -> 1)
    interval_idx <- as.integer(sub("int_003_edges.csv", "", sub("int_", "", sub(".csv", "", fname))))
    if(is.na(interval_idx)) interval_idx <- 1 # Fallback
    
    # 1. Compute Graph Metrics
    g_feats <- compute_graph_metrics(fpath)
    
    # 2. Get Raw Stats (Duration, Pressure)
    raw_feats <- get_raw_stats(CFG$base_dir, participant, gesture, interval_idx)
    
    # Combine
    row_data <- c(
      list(participant = participant, gesture = gesture),
      as.list(raw_feats),
      g_feats
    )
    results[[length(results)+1]] <- row_data
  }
  
  # Bind to Dataframe
  final_df <- bind_rows(results)
  
  # Clean NAs
  final_df[is.na(final_df)] <- 0
  
  dir.create(CFG$output_dir, recursive = TRUE, showWarnings = FALSE)
  out_file <- sprintf("%s/%s.csv", CFG$output_dir, CFG$input_subdir)
  write.csv(final_df, out_file, row.names = FALSE)
  msg("Done! Features saved to %s", out_file)
}

process_all()