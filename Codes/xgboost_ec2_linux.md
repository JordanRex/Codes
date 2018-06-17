scl enable devtoolset-2 bash

Test the environment:
  
gcc --version

gcc 4.8.x

source /opt/rh/devtoolset-2/enable

install.packages("RcppArmadillo", lib = "/home/user/R/x86_64-redhat-linux-gnu-library/3.3")


packageurl <- "https://cran.r-project.org/src/contrib/Archive/xgboost/xgboost_0.4-4.tar.gz"

packageurl <- "https://cran.r-project.org/src/contrib/Archive/RcppArmadillo/RcppArmadillo_0.6.100.0.0.tar.gz"
install.packages(packageurl, repos=NULL, type="source")