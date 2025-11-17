# build_interval_networks.R
# Creates one directed network per time interval by matching blobs (active-cell components)
# across consecutive frames with IoU > threshold, then wiring cells in matched blobs t -> t+1.
# Requires: igraph
library(igraph)

# ---------------------- Configuration (can be overridden via CLI) ----------------------
CFG <- list(
  base_dir        = "./../PressureSensorPi/",            # project root (contains ./harsh_process and ./.data or ./data)
  participant     = "ParticipantX", # e.g., "Anuradha_Right"
  gesture         = "SwipeV",       # e.g., "SwipeV"
  iou_threshold   = 0.20,           # IoU threshold for blob matching
  edge_mode       = "complete",     # "complete" = all pairs in A×B; "intersection" = only intersecting cells
  out_dir         = "networks",     # where to save PNGs/CSVs; created if missing
  plot_png        = TRUE,           # save plots per-interval as PNG
  png_width       = 1200,
  png_height      = 900,
  png_pointsize   = 14
)

# ---------------------- Utilities ----------------------
msg <- function(...) cat(sprintf(...), "\n")

# Return the best path that exists for timestamp file
resolve_timestamp_file <- function(base_dir, participant, gesture) {
  candidates <- c(
    file.path(base_dir, ".data", participant, sprintf("%s_timestamp_merge.txt", gesture)),
    file.path(base_dir, "data",  participant, sprintf("%s_timestamp_merge.txt", gesture)),
    file.path(base_dir, ".data", participant, sprintf("%s_timestamp.txt",        gesture)),
    file.path(base_dir, "data",  participant, sprintf("%s_timestamp.txt",        gesture))
  )
  existing <- candidates[file.exists(candidates)]
  if (length(existing) == 0) {
    stop("Could not find any timestamp file for gesture='", gesture,
         "' participant='", participant, "' in .data/ or ./data/", call. = FALSE)
  }
  existing[[1]]
}

# Parse intervals like "53 : 87" or "53,87" or "53 87" or "f0053 - f0087"
parse_intervals <- function(ts_path) {
  lines <- readLines(ts_path, warn = FALSE)
  keep   <- character()
  for (ln in lines) {
    ln <- trimws(ln)
    if (nzchar(ln) && !startsWith(ln, "#")) keep <- c(keep, ln)
  }
  if (length(keep) == 0) return(list())
  
  intervals <- list()
  for (ln in keep) {
    nums <- regmatches(ln, gregexpr("\\d+", ln))[[1]]
    if (length(nums) == 0) next
    if (length(nums) == 1) {
      a <- as.integer(nums[1]); b <- a
    } else {
      a <- as.integer(nums[1]); b <- as.integer(nums[2])
    }
    if (is.na(a) || is.na(b)) next
    if (b < a) { tmp <- a; a <- b; b <- tmp }
    intervals[[length(intervals)+1]] <- c(a, b)
  }
  intervals
}

# Read a frame matrix from harsh_process path. Returns numeric matrix.
read_frame_matrix <- function(base_dir, participant, gesture, frame_idx) {
  fn <- file.path(base_dir, "harsh_process", participant, gesture, sprintf("f%05d.csv", frame_idx))
  if (!file.exists(fn)) return(NULL)
  mat <- tryCatch({
    as.matrix(utils::read.csv(fn, header = FALSE, check.names = FALSE))
  }, error = function(e) NULL)
  pool_2x2(mat)
}

# Perform 2x2 max pooling. If dimensions are odd, the last row/column is dropped.
pool_2x2 <- function(mat) {
  nr <- nrow(mat); nc <- ncol(mat)
  if (is.null(nr) || is.null(nc) || nr < 2 || nc < 2) return(mat)
  
  r1 <- seq(1, nr - 1, by = 2)
  c1 <- seq(1, nc - 1, by = 2)
  pooled <- matrix(0, nrow = length(r1), ncol = length(c1))
  
  for (i in seq_along(r1)) {
    for (j in seq_along(c1)) {
      block <- mat[r1[i]:(r1[i] + 1L), c1[j]:(c1[j] + 1L)]
      pooled[i, j] <- max(block, na.rm = TRUE)
    }
  }
  
  pooled
}

# Label 4-neighbor connected components on a logical matrix; return list of integer-index vectors (linear indices)
label_components_4n <- function(mask) {
  nr <- nrow(mask); nc <- ncol(mask)
  if (is.null(nr) || is.null(nc) || nr == 0 || nc == 0) return(list())
  
  lab <- matrix(0L, nr, nc)
  comps <- vector("list", 0)
  lab_id <- 0L
  
  # neighbors: up, down, left, right
  nbors <- matrix(c(-1,0, 1,0, 0,-1, 0,1), ncol = 2, byrow = TRUE)
  
  for (r in 1:nr) {
    for (c in 1:nc) {
      if (!mask[r, c] || lab[r, c] != 0L) next
      lab_id <- lab_id + 1L
      
      # simple queue BFS
      qr <- integer(nr*nc); qc <- integer(nr*nc); head <- 1L; tail <- 1L
      qr[tail] <- r; qc[tail] <- c; tail <- tail + 1L
      lab[r, c] <- lab_id
      members <- integer(0)
      
      while (head < tail) {
        rr <- qr[head]; cc <- qc[head]; head <- head + 1L
        members <- c(members, (cc - 1L) * nr + rr) # linear index in column-major (R default)
        for (k in 1:nrow(nbors)) {
          r2 <- rr + nbors[k, 1]
          c2 <- cc + nbors[k, 2]
          if (r2 < 1L || r2 > nr || c2 < 1L || c2 > nc) next
          if (!mask[r2, c2] || lab[r2, c2] != 0L) next
          lab[r2, c2] <- lab_id
          qr[tail] <- r2; qc[tail] <- c2; tail <- tail + 1L
        }
      }
      comps[[lab_id]] <- members
    }
  }
  comps
}

# Compute IoU of two integer index sets (linear indices)
iou_sets <- function(a, b) {
  if (length(a) == 0 || length(b) == 0) return(0)
  inter <- length(intersect(a, b))
  if (inter == 0) return(0)
  uni   <- length(unique(c(a, b)))
  inter / uni
}

# Greedy one-to-one matching by IoU > threshold: pick the largest IoU first
match_components_by_iou <- function(comps_t, comps_t1, thresh) {
  if (length(comps_t) == 0 || length(comps_t1) == 0) return(list())
  pairs <- list()
  # Precompute IoUs
  ious <- matrix(0, nrow = length(comps_t), ncol = length(comps_t1))
  for (i in seq_along(comps_t)) {
    for (j in seq_along(comps_t1)) {
      ious[i, j] <- iou_sets(comps_t[[i]], comps_t1[[j]])
    }
  }
  # Greedy selection
  used_i <- rep(FALSE, length(comps_t))
  used_j <- rep(FALSE, length(comps_t1))
  repeat {
    # find max IoU among unused
    iou_max <- -1; best_i <- NA; best_j <- NA
    for (i in seq_along(comps_t)) if (!used_i[i]) {
      for (j in seq_along(comps_t1)) if (!used_j[j]) {
        val <- ious[i, j]
        if (!is.na(val) && val > iou_max) { iou_max <- val; best_i <- i; best_j <- j }
      }
    }
    if (is.na(best_i) || is.na(best_j) || iou_max < thresh) break
    pairs[[length(pairs)+1]] <- c(best_i, best_j)
    used_i[best_i] <- TRUE
    used_j[best_j] <- TRUE
  }
  pairs
}

# Build edges between two components, either complete bipartite or intersection-only
edges_between_components <- function(compA, compB, mode = "complete") {
  if (length(compA) == 0 || length(compB) == 0) return(matrix(integer(0), ncol = 2))
  if (mode == "intersection") {
    inter <- intersect(compA, compB)
    if (length(inter) == 0) return(matrix(integer(0), ncol = 2))
    # t -> t+1 edges only for cells that persist (same linear index)
    return(cbind(inter, inter))
  }
  # complete bipartite A x B
  as.matrix(expand.grid(compA, compB, KEEP.OUT.ATTRS = FALSE))
}

# Make graph from edge list (two-column integer matrix), attach grid layout coords
graph_from_edges_with_grid_layout <- function(edges, nr, nc) {
  # Ensure the graph always contains every vertex in the nr x nc grid
  all_ids <- seq_len(nr * nc)
  rc <- arrayInd(all_ids, .dim = c(nr, nc))
  Vtbl <- data.frame(
    name = as.character(all_ids),
    row  = rc[,1],
    col  = rc[,2],
    stringsAsFactors = FALSE
  )
  
  if (!is.null(edges) && nrow(edges) > 0) {
    Edf <- data.frame(
      from = as.character(edges[,1]),
      to   = as.character(edges[,2]),
      stringsAsFactors = FALSE
    )
  } else {
    Edf <- data.frame(from = character(), to = character(), stringsAsFactors = FALSE)
  }
  
  g <- graph_from_data_frame(Edf, directed = TRUE, vertices = Vtbl)
  # store a default layout in vertex attributes (x: col, y: flipped row so origin at bottom-left)
  V(g)$x <- V(g)$col
  V(g)$y <- nr - V(g)$row + 1
  g
}

plot_interval_graph <- function(g, title, save_png = FALSE, png_file = NULL,
                                width = 1200, height = 900, pointsize = 14) {
  if (vcount(g) == 0) {
    msg("[plot] %s — empty graph (no vertices)", title)
    return(invisible())
  }
  coords <- cbind(V(g)$x, V(g)$y)
  active_vertices <- degree(g, mode = "all") > 0
  vertex_colors <- ifelse(active_vertices, "black", NA)
  vertex_frames <- ifelse(active_vertices, "black", "gray70")
  if (save_png && !is.null(png_file)) {
    dir.create(dirname(png_file), recursive = TRUE, showWarnings = FALSE)
    png(filename = png_file, width = width, height = height, pointsize = pointsize)
  }
  op <- par(mar = c(1.5, 1.5, 3, 1))
  plot(
    g,
    layout = coords,
    vertex.size = 4,
    vertex.label = NA,
    vertex.color = vertex_colors,
    vertex.frame.color = vertex_frames,
    edge.arrow.size = 0.25,
    edge.curved = 0,
    main = title
  )
  par(op)
  if (save_png && !is.null(png_file)) dev.off()
}

# ---------- Discovery helpers (participants & gestures) ----------
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
  
  # keep only gestures that have some frames under harsh_process/<participant>/<gesture>
  keep <- vapply(gestures, function(g) {
    hp <- file.path(base_dir, "harsh_process", participant, g)
    if (!dir.exists(hp)) return(FALSE)
    any(grepl("^f\\d{4,5}\\.csv$", list.files(hp)))
  }, logical(1))
  
  sort(gestures[keep])
}

run_one_pair <- function(base_cfg, participant, gesture) {
  cfg <- base_cfg
  cfg$participant <- participant
  cfg$gesture     <- gesture
  msg("=== %s / %s ===", participant, gesture)
  tryCatch({
    build_networks_for_intervals(cfg)
    TRUE
  }, error = function(e) {
    msg("!! Skipped %s / %s — %s", participant, gesture, conditionMessage(e))
    FALSE
  })
}

# Batch driver:
# - If `participants` is NULL -> discover from .data/ and data/
# - If `participants` is a single name and `gestures` is NULL -> discover all gestures for that participant
# - If both provided -> iterate their cross product (only those with timestamp+frames will run)
build_all_networks <- function(cfg = CFG, participants = NULL, gestures = NULL) {
  if (is.null(participants)) {
    participants <- list_participants(cfg$base_dir)
  }
  if (!length(participants)) {
    stop("No participants found under .data/ or data/.", call. = FALSE)
  }
  
  total <- 0L; ok <- 0L
  for (p in participants) {
    G <- if (is.null(gestures)) list_gestures(cfg$base_dir, p) else gestures
    if (!length(G)) {
      msg("[info] No gestures found for participant '%s' — skipping.", p)
      next
    }
    for (g in G) {
      total <- total + 1L
      ok <- ok + as.integer(run_one_pair(cfg, p, g))
    }
  }
  msg("=== Done: %d/%d successful pair(s). ===", ok, total)
  invisible(ok == total)
}

# ---------------------- Main driver ----------------------
build_networks_for_intervals <- function(cfg = CFG) {
  ts_path <- resolve_timestamp_file(cfg$base_dir, cfg$participant, cfg$gesture)
  msg("Using timestamp file: %s", ts_path)
  
  intervals <- parse_intervals(ts_path)
  if (length(intervals) == 0) {
    stop("No valid intervals found in timestamp file.", call. = FALSE)
  }
  msg("Found %d interval(s).", length(intervals))
  
  # probe one frame to learn dimensions; fall back to 40x40 if not found
  probe <- NULL
  for (iv in intervals) {
    for (t in seq.int(iv[1], iv[2])) {
      probe <- read_frame_matrix(cfg$base_dir, cfg$participant, cfg$gesture, t)
      if (!is.null(probe)) break
    }
    if (!is.null(probe)) break
  }
  if (is.null(probe)) {
    stop("Could not read any frame CSV in harsh_process/…/fNNNN.csv for the listed intervals.", call. = FALSE)
  }
  nr <- nrow(probe); nc <- ncol(probe)
  msg("Detected frame dimensions: %d x %d (rows x cols).", nr, nc)
  
  # cache components per frame (avoid recomputation when frames reused)
  comp_cache <- new.env(parent = emptyenv())
  
  get_components <- function(frame_idx) {
    key <- as.character(frame_idx)
    if (!exists(key, envir = comp_cache, inherits = FALSE)) {
      mat <- read_frame_matrix(cfg$base_dir, cfg$participant, cfg$gesture, frame_idx)
      if (is.null(mat)) {
        assign(key, list(), envir = comp_cache); return(list())
      }
      mask <- (mat > 0)
      comps <- label_components_4n(mask)
      assign(key, comps, envir = comp_cache)
    }
    get(key, envir = comp_cache, inherits = FALSE)
  }
  
  # Ensure output directories
  iou_dir <- sprintf("%s_pool", cfg$iou_threshold)
  out_png_dir <- file.path(cfg$out_dir, iou_dir, cfg$participant, cfg$gesture, "png")
  out_csv_dir <- file.path(cfg$out_dir, iou_dir, cfg$participant, cfg$gesture, "csv")
  dir.create(out_png_dir, recursive = TRUE, showWarnings = FALSE)
  dir.create(out_csv_dir, recursive = TRUE, showWarnings = FALSE)
  
  # For each interval build edges by traversing t -> t+1
  for (k in seq_along(intervals)) {
    iv <- intervals[[k]]
    start_f <- iv[1]; end_f <- iv[2]
    msg("Interval %d: %d -> %d", k, start_f, end_f)
    
    all_edges <- list()
    edge_count <- 0L
    
    if (end_f > start_f) {
      for (t in seq.int(start_f, end_f - 1L)) {
        comps_t  <- get_components(t)
        comps_t1 <- get_components(t + 1L)
        if (length(comps_t) == 0 || length(comps_t1) == 0) next
        
        matches <- match_components_by_iou(comps_t, comps_t1, cfg$iou_threshold)
        if (length(matches) == 0) next
        
        for (mp in matches) {
          a <- comps_t[[mp[1]]]
          b <- comps_t1[[mp[2]]]
          eb <- edges_between_components(a, b, mode = cfg$edge_mode)
          if (nrow(eb)) {
            all_edges[[length(all_edges)+1]] <- eb
            edge_count <- edge_count + nrow(eb)
          }
        }
      }
    }
    
    if (length(all_edges) == 0) {
      msg("Interval %d produced 0 edges (empty network).", k)
      edges_mat <- matrix(integer(0), ncol = 2)
    } else {
      edges_mat <- do.call(rbind, all_edges)
    }
    
    g <- graph_from_edges_with_grid_layout(edges_mat, nr, nc)
    
    active_count <- sum(degree(g, mode = "all") > 0)
    
    title <- sprintf("%s / %s — interval %d [%d → %d], |V|=%d (%d active), |E|=%d, IoU≥%.2f, mode=%s",
                     cfg$participant, cfg$gesture, k, start_f, end_f, vcount(g), active_count,
                     ecount(g), cfg$iou_threshold, cfg$edge_mode)
    
    # Save edge list for the interval (optional but handy for R analysis)
    csv_path <- file.path(out_csv_dir, sprintf("%s_%s_interval_%03d_edges.csv",
                                               cfg$participant, cfg$gesture, k))
    if (ecount(g) > 0) {
      edge_df <- data.frame(
        from = ends(g, E(g))[,1],
        to   = ends(g, E(g))[,2],
        stringsAsFactors = FALSE
      )
    } else {
      edge_df <- data.frame(from = character(), to = character(), stringsAsFactors = FALSE)
    }
    utils::write.csv(edge_df, file = csv_path, row.names = FALSE)
    
    # Plot
    png_path <- file.path(out_png_dir, sprintf("%s_%s_interval_%03d.png",
                                               cfg$participant, cfg$gesture, k))
    plot_interval_graph(
      g, title,
      save_png   = cfg$plot_png,
      png_file   = png_path,
      width      = cfg$png_width,
      height     = cfg$png_height,
      pointsize  = cfg$png_pointsize
    )
    msg("  -> saved: %s", png_path)
    msg("  -> edges CSV: %s", csv_path)
  }
  
  invisible(TRUE)
}


# ---------------------- RUN ----------------------
build_all_networks(CFG)
