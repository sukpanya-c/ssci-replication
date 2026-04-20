parse_args <- function(args) {
  out <- list(
    draws = 5000,
    burnin = 1000,
    thin = 10,
    horizons = 12,
    dates = NULL
  )
  i <- 1
  while (i <= length(args)) {
    key <- args[[i]]
    val <- if (i < length(args)) args[[i + 1]] else NA_character_
    if (key == "--input") out$input <- val
    if (key == "--output") out$output <- val
    if (key == "--draws") out$draws <- as.integer(val)
    if (key == "--burnin") out$burnin <- as.integer(val)
    if (key == "--thin") out$thin <- as.integer(val)
    if (key == "--horizons") out$horizons <- as.integer(val)
    if (key == "--dates") out$dates <- strsplit(val, ",", fixed = TRUE)[[1]]
    i <- i + 2
  }
  if (is.null(out$input) || is.null(out$output)) {
    stop("Usage: Rscript run_tvp_svar.R --input <csv> --output <dir> [--draws N --burnin N --thin N --horizons H --dates d1,d2]")
  }
  out
}

summarize_irf <- function(irf_draws, contemporaneous, horizons, meta) {
  horizon_values <- c(0, seq_len(horizons))
  qfun <- function(x) unname(stats::quantile(x, probs = c(0.05, 0.25, 0.5, 0.75, 0.95), na.rm = TRUE))
  quants <- lapply(horizon_values, function(h) {
    draws <- if (h == 0) contemporaneous else irf_draws[, h]
    qfun(draws)
  })
  frame <- data.frame(
    horizon = horizon_values,
    lower_90 = vapply(quants, `[[`, numeric(1), 1),
    lower_50 = vapply(quants, `[[`, numeric(1), 2),
    median = vapply(quants, `[[`, numeric(1), 3),
    upper_50 = vapply(quants, `[[`, numeric(1), 4),
    upper_90 = vapply(quants, `[[`, numeric(1), 5)
  )
  for (nm in names(meta)) {
    frame[[nm]] <- meta[[nm]]
  }
  frame
}

args <- parse_args(commandArgs(trailingOnly = TRUE))

if (!requireNamespace("bvarsv", quietly = TRUE)) {
  stop("Package 'bvarsv' is required.")
}

dir.create(args$output, recursive = TRUE, showWarnings = FALSE)
input <- utils::read.csv(args$input, stringsAsFactors = FALSE)
if (!"Date" %in% names(input)) {
  stop("Input file must include a Date column.")
}

input$Date <- as.Date(input$Date)
vars <- c("GPR_ME", "oil_shock", "fx_dep", "inflation_mom")
missing_vars <- setdiff(vars, names(input))
if (length(missing_vars) > 0) {
  stop(sprintf("Missing required columns: %s", paste(missing_vars, collapse = ", ")))
}

model_data <- input[, vars]
complete_rows <- stats::complete.cases(model_data)
model_data <- model_data[complete_rows, , drop = FALSE]
input <- input[complete_rows, , drop = FALSE]
tau <- min(40, max(12, floor(nrow(model_data) / 3)))
p <- 1

set.seed(7)
fit <- bvarsv::bvar.sv.tvp(
  as.matrix(model_data),
  p = p,
  tau = tau,
  nrep = args$draws,
  nburn = args$burnin,
  thinfac = args$thin,
  save.parameters = TRUE,
  itprint = max(args$draws, 1000)
)

if (is.null(args$dates)) {
  args$dates <- tail(as.character(input$Date), 3)
}

pairs <- list(
  list(impulse = "GPR_ME", response = "oil_shock"),
  list(impulse = "oil_shock", response = "fx_dep"),
  list(impulse = "oil_shock", response = "inflation_mom"),
  list(impulse = "fx_dep", response = "inflation_mom")
)

irf_rows <- list()
for (date_str in args$dates) {
  idx <- which(input$Date == as.Date(date_str))
  if (length(idx) == 0) next
  t_index <- idx[[1]] - (tau + p)
  if (t_index < 1 || t_index > dim(fit$Beta.draws)[2]) next
  for (pair in pairs) {
    irf <- bvarsv::impulse.responses(
      fit,
      impulse.variable = match(pair$impulse, vars),
      response.variable = match(pair$response, vars),
      t = t_index,
      nhor = args$horizons,
      scenario = 2,
      draw.plot = FALSE
    )
    irf_rows[[length(irf_rows) + 1]] <- summarize_irf(
      irf$irf,
      irf$contemporaneous,
      args$horizons,
      list(
        date = date_str,
        impulse = pair$impulse,
        response = pair$response,
        scenario = "cholesky"
      )
    )
  }
}

if (length(irf_rows) == 0) {
  stop("No requested dates matched the input sample.")
}

irf_summary <- do.call(rbind, irf_rows)
utils::write.csv(irf_summary, file.path(args$output, "tvp_svar_irf_summary.csv"), row.names = FALSE)

fit_summary <- data.frame(
  metric = c("nobs", "draws", "burnin", "thin", "horizons", "tau"),
  value = c(nrow(model_data), args$draws, args$burnin, args$thin, args$horizons, tau)
)
utils::write.csv(fit_summary, file.path(args$output, "tvp_svar_fit_summary.csv"), row.names = FALSE)
