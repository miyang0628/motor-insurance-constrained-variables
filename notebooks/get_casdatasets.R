
Sys.setenv(R_DEFAULT_INTERNET_TIMEOUT = "600")
options(timeout = 600)
options(download.file.method = "libcurl")

user_lib <- file.path(Sys.getenv("USERPROFILE"), "R", "win-library", "4.6")
dir.create(user_lib, recursive = TRUE, showWarnings = FALSE)
.libPaths(c(user_lib, .libPaths()))

if (!requireNamespace("xts", quietly = TRUE)) {
  install.packages("xts", repos = "https://cloud.r-project.org", lib = user_lib)
}

if (!requireNamespace("CASdatasets", quietly = TRUE)) {
  install.packages(
    "CASdatasets",
    repos = "https://dutangc.perso.math.cnrs.fr/RRepository/pub/",
    type  = "source",
    lib   = user_lib
  )
}

library(CASdatasets, lib.loc = user_lib)
data(fremotor2freq9907b)
write.csv(fremotor2freq9907b, "data/fremotor2freq9907b.csv", row.names = FALSE)
cat("fremotor2freq9907b saved.\n")
