# devtools::install_github('petarstf/ccdefault')
library(opencpu)
# detach('package:ccdefault', unload = T)
# devtools::install_local('D:/IBIS/ccdefault', force = T)
library(ccdefault)

# opencpu::ocpu_start_server(port = 8001, workers = 6, preload = c('opencpu', 'ccdefault'))
opencpu::ocpu_start_app('ccdefault', port = 8001, workers = 6)
