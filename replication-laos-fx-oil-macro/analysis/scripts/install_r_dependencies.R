packages <- c("bvarsv")

missing_packages <- packages[!vapply(packages, requireNamespace, logical(1), quietly = TRUE)]

if (length(missing_packages) > 0) {
  install.packages(missing_packages, repos = "https://cloud.r-project.org")
}

versions <- vapply(
  packages,
  function(pkg) as.character(utils::packageVersion(pkg)),
  character(1)
)

for (pkg in packages) {
  cat(sprintf("%s %s\n", pkg, versions[[pkg]]))
}
