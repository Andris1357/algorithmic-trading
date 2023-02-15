install.packages("devtools"); devtools::install_github("dmurdoch/rgl"); install.packages("dplyr"); install.packages("roperators")
library(rgl); library(dplyr); library(glue); library(barplot3d); require(roperators)

`%?%` <- function(evaluate__vc, condition) {
   if(!condition) {return(evaluate__vc[[2]])}
   else {return(evaluate__vc[[1]])}
}
`%:%` <- function(on_true, on_false) {
   return(list(on_true, on_false))
}

df_calc2 <- read.table("klines_all_3 (1).csv", sep = ",", dec = ".", quote = "", col.names = c("ticker", "datetime", "close", "volume"))
pairs <- unique(df_calc2$ticker)

f_get_profit <- function (pairs_in = pairs, input = df_calc2) {
   pairs_in <- pairs_in[!pairs_in %in% c("BUSDUSDT", "PAXUSDT", "TUSDUSDT", "USDCUSDT", "EURUSDT", "GBPUSDT", "AUDUSDT", "SUSDUSDT", "COCOSUSDT")]
   df_ls <- append(list(), c(rep(0, length(pairs_in))))
   for (i in (1: length(pairs_in))) {
      df_i <- filter(input, ticker == pairs_in[i]) %>% select(close, volume)
      if (dim(df_i)[1] < 48) {print(glue("{pairs_in[i]} skipped")); next}
      returns_i <- sapply(c(1: (dim(df_i)[1] - 47)), function(y) max(df_i$close[c((y + 1) : (y + 47))]) / df_i$close[y] - 1)
      df_ls[[i]] <- append(df_ls[[i]], returns_i)
      df_ls[[i]] <- df_ls[[i]][c(2: length(df_ls[[i]]))]
   }
   return (df_ls)
}
df_ls_out <- f_get_profit()

f_find_pattern <- function(vals) {
   vals <- filter(df_calc2, ticker == vals) %>% select(close)
   for (i in (1: (dim(vals)[1] - 100))) {
      for (i2 in ((i + 1) : (dim(vals)[1]))) {
         if (all(vals[c(i2: (i2 + 100)), 1] == vals[c(i: (i + 100)), 1])) {
            print(paste("start1: ", i, " start2: ", i2))
            return (c(i, i2))}
      }
   }
}

#find pattern in all pricelists, remove dupl vals -> fill shorter vector outstanding spaces w 0s => graph
f_clean_data <- function(pairs_in) {
   slice_ls <- append(list(), c(rep(0, length(pairs_in))))
   for (pair_i in (1: (dim(pairs_in)[1]))) {
      slice <- f_find_pattern(pairs_in[pair_i])
      slice_ls[pair_i] <- append(slice_ls[pair_i], slice)
   }
   most_length <- sum(sapply(sapply(df_ls, function(y) length(y)), function(y2) 1 %:% 0 %?% (y2 > 9824)))
}

# I ARRIVED AT MOST RELEVANT LENTGH = 9871 WITH THE HELP OF ABOVE 2 FUNCTIONS

f_get_dists <- function(pair, pairs_in, df_ls_in, uniform_length_in, dims, type2d = "none", w_lists = TRUE) {
   df_len <- length(df_ls_in[[which(pairs_in == pair)]]); if (df_len > 9871) {df_len <- 9871}
   if (df_len < 48) {return ("invalid")}
   data <- data.frame(sapply(1:4, function (y) c(rep(0, df_len)))) 
   data[, 1] <- df_ls_in[[which(pairs_in == pair)]] 
   data[, c(2,3)] <- filter(df_calc2, ticker == pair) %>% select(close, volume) %>% (function (y) y[c(1:df_len), ])()
   data[c(2: (dim(data)[1])), 4] <- sapply(c(2: (dim(data)[1])), function (y) (data[y, 2] - data[(y - 1), 2]) / data[(y - 1), 2])
   data <- data[-1, ] # because we cannot measure some params for the first candle
   chg_bounds <- c(min(data[, 4]), max(data[, 4])); pr_bounds <- c(min(data[, 2]), max(data[, 2]))
   vol_bounds <- c(min(data[, 3]), max(data[, 3])); print(glue("bounds\nchg: {chg_bounds}\nvol: {vol_bounds}\npr: {pr_bounds}"))
   
   if (!missing(uniform_length_in)) {
      print(glue("Uniform length {uniform_length_in} used"))
      chg_scale <- sapply(1: (uniform_length_in + 1), function (y) chg_bounds[1] + (y - 1) * (chg_bounds[2] - chg_bounds[1]) / uniform_length_in)
      vol_scale <- sapply(1: (uniform_length_in + 1), function (y) vol_bounds[1] + (y - 1) * (vol_bounds[2] - vol_bounds[1]) / uniform_length_in)
      pr_scale <- sapply(1: (uniform_length_in + 1), function (y) pr_bounds[1] + (y - 1) * (pr_bounds[2] - pr_bounds[1]) / uniform_length_in)
      print(glue("{length(chg_scale)}, {length(vol_scale)}, {length(pr_scale)}"))
   }
   else {
      print(glue("Individual lengths {length(data[, 2])} (pr), {length(data[, 3])} (vol) {length(data[, 4])} (chg) used"))
      chg_scale <- sapply(1: (length(data[, 4]) + 1), function (y) chg_bounds[1] + (y - 1) * (chg_bounds[2] - chg_bounds[1]) / length(data[, 4]))
      vol_scale <- sapply(1: (length(data[, 3]) + 1), function (y) vol_bounds[1] + (y - 1) * (vol_bounds[2] - vol_bounds[1]) / length(data[, 3]))
      pr_scale <- sapply(1: (length(data[, 2]) + 1), function (y) pr_bounds[1] + (y - 1) * (pr_bounds[2] - pr_bounds[1]) / length(data[, 2]))
   }
   
   chg_ret_dist <- sapply(1:length(chg_scale), function (y) 0)
   vol_ret_dist <- sapply(1:length(vol_scale), function (y) 0)
   pr_ret_dist <- sapply(1:length(vol_scale), function (y) 0)
   assign_gate <- function (lowerbound, val, upperbound) lowerbound %:% upperbound %?% (abs(val - lowerbound) < abs(val - upperbound))
   if (dims == 2 & type2d != "none") {
      par_ids <- strsplit(type2d, "/")[[1]]; print(par_ids)
      par1_scale <- list(chg_scale, vol_scale, pr_scale) %>% (function (x) x[[which(c("ch", "vo", "pr") == substr(par_ids[1], 1, 2))]])
      par2_scale <- list(chg_scale, vol_scale, pr_scale) %>% (function (x) x[[which(c("ch", "vo", "pr") == substr(par_ids[2], 1, 2))]])
      if (w_lists == TRUE) {p1p2_ls <- lapply(1:length(par1_scale), function (x) c(rep(0, length(par2_scale))))}
      else {p1p2_mx <- matrix(rep(0, length(par1_scale) * length(par2_scale)), nrow = length(par1_scale), ncol = length(par2_scale))}
   }
   else if (dims == 3) {
      space4d <- array(c(rep(0, (uniform_length_in + 1) ^ 3)), c(uniform_length_in + 1, uniform_length_in + 1, uniform_length_in + 1))
   }
   
   for (i in 1:length(data[, 1])) { #ADD ONE CANDLE'S CHARACTERISTICS TO THE DISTRIBUTIONS
      ret_i <- data[i, 1] ; chg_i <- data[i, 4] ; vol_i <- data[i, 3] ; pr_i <- data[i, 2]
      chg_higher_neighbor <- chg_scale[which(chg_scale >= chg_i)][1]
      chg_lower_neighbor <- chg_scale[which(chg_scale <= chg_i)][length(chg_scale[which(chg_scale <= chg_i)])]
      vol_higher_neighbor <- vol_scale[which(vol_scale >= vol_i)][1]
      vol_lower_neighbor <- vol_scale[which(vol_scale <= vol_i)][length(vol_scale[which(vol_scale <= vol_i)])]
      pr_higher_neighbor <- pr_scale[which(pr_scale >= pr_i)][1]
      pr_lower_neighbor <- pr_scale[which(pr_scale <= pr_i)][length(pr_scale[which(pr_scale <= pr_i)])]
      
      if (dims == 2 & type2d != "none") { #constructs cross-distribution
         neighbors <- c(chg_lower_neighbor, vol_lower_neighbor, 
                        pr_lower_neighbor, chg_i, chg_higher_neighbor, vol_i, vol_higher_neighbor, pr_i, pr_higher_neighbor); print(neighbors)
         names(neighbors) <- c("chg_lower_neighbor", "vol_lower_neighbor", 
                        "pr_lower_neighbor", "chg_i", "chg_higher_neighbor", "vol_i", "vol_higher_neighbor", "pr_i", "pr_higher_neighbor")
         pars1 <- neighbors[which(substr(names(neighbors), 1, 2) == substr(par_ids[1], 1, 2))] 
         pars2 <- neighbors[which(substr(names(neighbors), 1, 2) == substr(par_ids[2], 1, 2))]
         stopifnot(length(which(c("ch", "vo", "pr") == substr(par_ids[1], 1, 2))) > 0)
         print(which(c("ch", "vo", "pr") == substr(par_ids[2], 1, 2)))
         par1_match <- assign_gate(pars1[1], pars1[2], pars1[3]); print(glue("par1 match: {par1_match}"))
         par2_match <- assign_gate(pars2[1], pars2[2], pars2[3]); print(glue("par2 match: {par2_match}"))
         split_cond <- c(pars2[3] - pars2[2] == pars2[2] - pars2[1], pars1[3] - pars1[2] == pars1[2] - pars1[1]); print(glue("split? {split_cond}"))
         if (any(split_cond == TRUE)) {
            neighbors_map <- list(pars1, pars2); print(paste("map:", neighbors_map))
            matches_nls <- lapply(1:2, function (x) neighbors_map[[x]][c(1,3)] %:% c(par1_match, par2_match)[x] %?% (split_cond[x] == TRUE))
         }
         
         if (w_lists == TRUE) {
            if (any(split_cond == TRUE)){
               if (all(split_cond == TRUE)) {
                  p1p2_ls[[matches_nls[[1]][1]]][matches_nls[[2]][1]] %+=% ret_i / 4; p1p2_ls[[matches_nls[[1]][2]]][matches_nls[[2]][1]] %+=% ret_i / 4
                  p1p2_ls[[matches_nls[[1]][1]]][matches_nls[[2]][2]] %+=% ret_i / 4; p1p2_ls[[matches_nls[[1]][2]]][matches_nls[[2]][2]] %+=% ret_i / 4
               }
               else if (split_cond[1] == TRUE) {
                  p1p2_ls[[matches_nls[[1]][1]]][par2_match] %+=% ret_i / 2; p1p2_ls[[matches_nls[[1]][2]]][par2_match] %+=% ret_i / 2
               }
               else {
                  p1p2_ls[[par1_match]][matches_nls[[2]][1]] %+=% ret_i / 2; p1p2_ls[[par1_match]][matches_nls[[2]][2]] %+=% ret_i / 2
               }
            }
            else {
               print(glue("index {c(which(par2_scale == par2_match), which(par1_scale == par1_match))}")); print(glue("prev value {p1p2_ls[[which(par1_scale == par1_match)]][which(par2_scale == par2_match)]}"))
               p1p2_ls[[which(par1_scale == par1_match)]][which(par2_scale == par2_match)] %+=% ret_i}
               print(glue("new value {p1p2_ls[[which(par1_scale == par1_match)]][which(par2_scale == par2_match)]}"))
            }
         else {
            if (any(split_cond == TRUE)) {
               if (all(split_cond == TRUE)) {p1p2_mx[c(which(par1_scale == matches_nls[[1]])), c(which(par1_scale == matches_nls[[1]]))] %+=% as.double(ret_i) / 4}
               else {p1p2_mx[which(par1_scale == matches_nls[[1]]), which(par1_scale == matches_nls[[1]])] %+=% as.double(ret_i) / 2}
            }
            else {
               print("mx data:"); print(c(which(par1_scale == par1_match), which(par2_scale == par2_match))); print(p1p2_mx[which(par1_scale == par1_match), which(par2_scale == par2_match)])
               p1p2_mx[which(par1_scale == par1_match), which(par2_scale == par2_match)] %+=% as.double(ret_i); print(p1p2_mx[which(par1_scale == par1_match), which(par2_scale == par2_match)])
            }
         }
      }
      
      else if (dims == 3) { #locate dot w/i arr
         pars_mx <- matrix(c(chg_lower_neighbor, chg_i, chg_higher_neighbor, vol_lower_neighbor, vol_i, vol_higher_neighbor, pr_lower_neighbor, pr_i, pr_higher_neighbor),
                           nrow = 3, ncol = 3) ; matches <- c()
         split_cond3 <- sapply(1:(dim(pars_mx)[1]), function (x) abs(pars_mx[1, x] - pars_mx[2, x]) == abs(pars_mx[3, x] - pars_mx[2, x]))
         if (!any(split_cond3)) {
            eq_matches <- sapply(1:3, function (x) assign_gate(pars_mx[1, x], pars_mx[2, x], pars_mx[3, x]))
            scales <- list(chg_scale, vol_scale, pr_scale)
            coords <- sapply(1:3, function (x) which(scales[[x]] == eq_matches[x]))
            space4d[coords[1], coords[2], coords[3]] %+=% as.double(ret_i)
         } else {
            div_factor <- 1 / 2 ^ length(which(split_cond3 == TRUE))
            space4d[which(chg_scale == (pars_mx[1, c(1,3)] %:% which(pars_mx[1, ] == assign_gate(pars_mx[1, 1], pars_mx[1, 2], pars_mx[1, 3])) %?% split_cond3[1])), 
                    which(vol_scale == (pars_mx[1, c(1,3)] %:% which(pars_mx[2, ] == assign_gate(pars_mx[2, 1], pars_mx[2, 2], pars_mx[2, 3])) %?% split_cond3[2])), 
                    which(pr_scale == (pars_mx[1, c(1,3)] %:% which(pars_mx[3, ] == assign_gate(pars_mx[3, 1], pars_mx[3, 2], pars_mx[3, 3])) %?% split_cond3[3]))
                    ] %+=% as.double(ret_i) / div_factor
         }
      }
      
      else if (dims == 1) {
         if (chg_i == 0) {chg_ret_dist[1] %+=% ret_i
         } else if (chg_i %in% chg_bounds) {chg_ret_dist[which(chg_scale == chg_i)] %+=% ret_i
         } else if (chg_higher_neighbor - chg_i > chg_i - chg_lower_neighbor) {
            chg_ret_dist[which(chg_scale == chg_lower_neighbor)] %+=% ret_i
         } else if (chg_higher_neighbor - chg_i == chg_i - chg_lower_neighbor) {
            chg_ret_dist[which(chg_scale == chg_higher_neighbor)] %+=% ret_i / 2
            chg_ret_dist[which(chg_scale == chg_lower_neighbor)] %+=% ret_i / 2
         } else {chg_ret_dist[which(chg_scale == chg_higher_neighbor)] %+=% ret_i}
         if (vol_i == 0) {vol_ret_dist[1] %+=% ret_i
         } else if (vol_i %in% vol_bounds) {vol_ret_dist[which(vol_scale == vol_i)] %+=% ret_i
         } else if (vol_higher_neighbor - vol_i > vol_i - vol_lower_neighbor) {vol_ret_dist[which(vol_scale == vol_lower_neighbor)] %+=% ret_i
         } else if (vol_higher_neighbor - vol_i == vol_i - vol_lower_neighbor) {
            vol_ret_dist[which(vol_scale == vol_higher_neighbor)] %+=% ret_i / 2
            vol_ret_dist[which(vol_scale == vol_lower_neighbor)] %+=% ret_i / 2
         } else {vol_ret_dist[which(vol_scale == vol_higher_neighbor)] %+=% ret_i}
         if (pr_higher_neighbor - pr_i == pr_i - pr_lower_neighbor) {
            pr_ret_dist[which(pr_scale == pr_higher_neighbor)] %+=% ret_i / 2}
      }
      
      else if (dims == 2 & type2d == "none") {return ("Must specify 2d axes")}
   }
   
   if (dims == 2) {
      if (w_lists == TRUE) {
         return (list(p1p2_ls, strsplit(type2d, "/"), par1_scale, par2_scale))
      }
      else {
         return (list(p1p2_mx, strsplit(type2d, "/"), par1_scale, par2_scale))}
      }
   else if (dims == 1) {
      if (type2d == "scale1d") {return (list(chg_scale, vol_scale, pr_scale))
      } else {return (list(chg_ret_dist, vol_ret_dist, pr_ret_dist))}}
   else if (dims == 3) {return (space4d)}
} 

#2DIM
dim1_dists <- f_get_dists("MATICUSDT", pairs, df_ls_out, 274, 1, "", F)
dim1_scales <- f_get_dists("MATICUSDT", pairs, df_ls_out, 274, 1, "scale1d", F)
barplot(dim1_dists[[1]])

#3DIM
terrain <- f_get_dists("MATICUSDT", pairs, df_ls_out, 
                             (sapply(df_ls_out, function (y) length(y)) %>% (function (y2) min(y2[which(y2 >= 50)]))() * 2), 
                             2, "chg/vol", FALSE) #creates matrix; INPUT: ccy, pars
mx_todf <- as.data.frame(terrain[[1]], row.names = terrain[[3]], optional = FALSE, make.names = TRUE, stringsAsFactors = default.stringsAsFactors()) %>%
   rbind(append(terrain[[4]], NA))
write.csv(mx_todf, "matic_chgvol.csv") #INPUT: filename

#4 DIM
space4 <- f_get_dists("MATICUSDT", pairs, df_ls_out, 
                       (sapply(df_ls_out, function (y) length(y)) %>% (function (y2) min(y2[which(y2 >= 50)]))() * 2), 
                      3, "none", FALSE) #creates array

arr3dToExcel <- function (chg_scale_in, vol_scale_in, pr_scale_in, arr) {
   space4_excel <- as.data.frame(arr[, , 1], row.names = chg_scale_in, optional = FALSE, make.names = TRUE, stringsAsFactors = default.stringsAsFactors()) %>%
      rbind(c(cat("PRICE LEVEL =", arr[, , 1]), rep("#", dim(arr)[2] - 1)))
   for (layer in 2:(dim(arr)[3])) {
      append_i <- as.data.frame(
        sapply(
          1:(dim(arr)[2] + 1), 
          function (x) chg_scale_in %:% arr[ , x, layer] %?% x == 1
        ), 
        row.names = NULL, optional = FALSE, make.names = FALSE, stringsAsFactors = default.stringsAsFactors()
      )
      colnames(append_i) <- vol_scale_in 
      space4_excel <- rbind(space4_excel, append_i) %>% rbind(c(cat("PRICE LEVEL =", arr[, , layer]), rep("#", dim(arr)[2] - 1)))
   }
   return (space4_excel)
}

arr_flat_df <- arr3dToExcel(dim1_scales[1], dim1_scales[2], dim1_scales[3], space4)
boundFunc <- bindLasts(arr3dToExcel, arr = space4)
write.csv(do.call("arr3dToExcel", as.list(dim1_scales)), "matic_linspace.csv")

write.csv(data.frame(matic_py_inputs[[1]], matic_py_inputs[[2]]), "matic_py_lookup.csv")
write.csv(as.data.frame(cbind(dim1_scales[[1]], dim1_scales[[2]]) %>% cbind(
  dim1_scales[[3]]), row.names = 1:length(dim1_scales[[1]]), 
  optional = FALSE, make.names = TRUE, stringsAsFactors = default.stringsAsFactors()), "matic_scales.csv")

bindLasts <- function(fun,...)
{
   boundArgs <- list(...) #puts all args into ls
   formals(fun)[names(boundArgs)] <- boundArgs
   fun
}
