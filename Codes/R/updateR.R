setInternet2(TRUE)
updateR()
devtools::install_github('talgalili/installr', force = TRUE)
readLines(url("http://www.google.co.in", method = "wininet"))

options(repos = c(CRAN = "http://cran.rstudio.com"))

chooseCRANmirror(graphics = getOption("menu.graphics"), ind = NULL,
                 useHTTPS = getOption("useHTTPS", TRUE),
                 local.only = FALSE)
