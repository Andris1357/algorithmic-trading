install.packages("dplyr")
install.packages("roperators")
install.packages("stringr")
library(stringr)
library(dplyr)
library(glue)
require(roperators)


getMode <- function(vector_) {
  tmp_distinct <- unique(vector_)
  return (tmp_distinct[which.max(tabulate(match(vector_, tmp_distinct)))])
}

# Removing tables that are likely to be a stablecoin due to their low standard deviation, thus would not be traded on
cleanTable <- function(table_, pair_) {
  table_ <- table_[table_$pair.symbol == pair_, ]
  table_ <- table_[, -2]
  if (sd(table_$close) < 0.05) {
    return (NULL)
  }
  else {
    return (table_)
  }
}

cleanData <- function(table_feature_timeseries_, pairs_) {
  options(digits=15)
  
  t_pair_lengths <- sapply(
    pairs_, 
    function (pair_) dim(
      table_feature_timeseries_[table_feature_timeseries_$pair.symbol == pair_, ]
    )[1]
  )
  t_mode_length <- getMode(t_pair_lengths)
  print(glue("Most prevalent length: {t_mode_length}"))
  
  table_feature_timeseries_$open <- as.double(gsub(
    "[\"\\'\\[]", "", table_feature_timeseries_$open, perl=TRUE
  ))
  table_feature_timeseries_$taker.buy.quote <- as.double(gsub(
    "[\\'\"\\]]", "", table_feature_timeseries_$taker.buy.quote, perl=TRUE
  ))
  for (column_i_ in c(4,5,6,7,9,11)) {
    table_feature_timeseries_[, column_i_] <- as.double(gsub(
      "'", "", table_feature_timeseries_[, column_i_]
    ))
  }
  # filter by those pairs that are greater than the mode
  table_feature_timeseries_ <- lapply(
    pairs_[which(t_pair_lengths >= t_mode_length)], 
    function (pair_) cleanTable(table_feature_timeseries_, pair_) # returns a list of dfs
  )
  table_feature_timeseries_ <- table_feature_timeseries_[which(
    !is.null(table_feature_timeseries_)
  )]
  
  return (table_feature_timeseries_)
}

calculateEMAWeights <- function(ema_length_) {
  return (1 : ema_length_ * (1 / (ema_length_ / 2 + 0.5)))
}

# Take into account UP, DOWN & other leveraged products
# Bollinger bands, logarithmic regression line -> over|undervalued
appendCalculatedMetrics <- function(table_) {
  t_lookback_lengths <- c(14, 21, 30, 50)
    
  t_rsi_change_buy_ls <- lapply(
    t_lookback_lengths, 
    function (rsi_length_) sapply(
      1: (dim(table_)[1] - rsi_length_ + 1),
      function (candle_i_) (function (change_vector_
        ) sum(change_vector_[which(change_vector_ >= 0)]) / sum(abs(change_vector_[which(change_vector_ < 0)]))
      )(table_$close[
        c(candle_i_ : (candle_i_ + rsi_length_ - 1))
      ] - table_$open[
        c(candle_i_ : (candle_i_ + rsi_length_ - 1))
      ])
    )
  )
  
  t_rsi_vol_buy_ls <- lapply(
    t_lookback_lengths, 
    function (rsi_length_) sapply(
      1: (dim(table_)[1] - rsi_length_ + 1),
      function (candle_i_) (function (change_vector_
        ) sum(table_$volume[which(change_vector_ >= 0)]) / sum(abs(change_vector_[which(change_vector_ < 0)]))
      )(table_$close[c(candle_i_ : (candle_i_ + rsi_length_ - 1))] - table_$open[
        c(candle_i_ : (candle_i_ + rsi_length_ - 1))
      ])
    )
  )
  
  t_sma_ls <- lapply(t_lookback_lengths, function (sma_length_) sapply(
    sma_length_: (dim(table_)[1]),
    function (candle_i_) mean(
      table_$close[
        c((candle_i_ - sma_length_) : candle_i_)
      ]
    )
  ))
  
  t_ema_ls <- lapply(t_lookback_lengths, function (ema_length_) sapply(
    1 : (dim(table_)[1] - ema_length_ + 1),
    function (candle_i_) sapply(
      1 : ema_length_, 
      function (weight_i_) sum(
        calculateEMAWeights(ema_length_) * table_$close[
          c(candle_i_ : (candle_i_ + ema_length_ - 1))
        ]
      ) / ema_length_
    )
  ))
  
  t_macd_ls <- lapply(
    2 : length(t_lookback_lengths),
    function (list_i_) t_sma_ls[[list_i_]] - t_sma_ls[[list_i_ - 1]]
  )
  
  t_average_change <- mean(sapply(
    2 : dim(table_)[1], 
    function (candle_i_) table_$close[candle_i_] / table_$close[candle_i_ - 1]
  ))
  
  t_hist_vol_ls <- lapply(
    t_lookback_lengths, function (hvol_length_) sapply(
      1: (dim(table_)[1] - hvol_length_), function (candle_i_) (sapply(
        (candle_i_ + 1) : (candle_i_ + hvol_length_),
        function (change_i_) (table_$close[change_i_] / table_$close[change_i_ - 1] - 1 - mean(sapply(
          (candle_i_ + 1) : (candle_i_ + hvol_length_),
          function (change_i2_) table_$close[change_i2_] / table_$close[change_i2_ - 1]
      ))) ^ 2 / (hvol_length_ - 1)) ^ 0.5
    ))
  )

  t_atr_ls <- lapply(t_lookback_lengths, function (atr_len) sapply(
    1: (dim(table_)[1] - atr_len),
    function (candle_i_) mean(abs(table_$close[
      c(candle_i_ : candle_i_ + atr_len)
    ] - table_$open[
      c(candle_i_ : candle_i_ + atr_len)
    ]))
  ))
  
  t_atr_extreme_ls <- lapply(t_lookback_lengths, function (atr_len) sapply(
    1: (dim(table_)[1] - atr_len),
    function (candle_i_) mean(abs(table_$high[
      c(candle_i_ : candle_i_ + atr_len)
    ] - table_$low[
      c(candle_i_ : candle_i_ + atr_len)
    ]))
  ))
  
  t_average_order_size <- sapply(
    1: dim(table_)[1], function (candle_i_) table_$volume / table_$trade.quantity
  )
  
  t_average_exchange_price <- table_$quote.volume / table_$volume
  
  t_taker_sell_quote <- table_$taker.buy.base * t_average_exchange_price
  t_maker_sell_base <- table_$taker.buy.base
  t_maker_buy_quote <- table_$taker.buy.base * t_average_exchange_price
  t_taker_sell_base <- table_$taker.buy.quote / t_average_exchange_price
  t_maker_sell_quote <- table_$taker.buy.quote
  t_maker_buy_base <- table_$taker.buy.quote / t_average_exchange_price
  
  t_maker_base <- table_$volume - t_taker_sell_base - table_$taker.buy.base
  t_maker_quote <- table_$quote.volume - t_taker_sell_quote - table_$taker.buy.quote
  t_taker_base <- table_$taker.buy.base + t_taker_sell_base
  t_taker_quote <- table_$taker.buy.quote + t_taker_sell_quote
  
  t_buy_base <- table_$taker.buy.base + t_maker_buy_base
  t_sell_base <- t_taker_sell_base + t_maker_sell_base
  
  t_buy_sell_ratio <- t_buy_base / t_sell_base
  t_taker_per_maker_base <- t_taker_base / t_maker_base
  t_taker_per_maker_quote <- t_taker_quote / t_maker_quote
  t_buyer_per_taker_base <- t_buy_base / t_taker_base
  t_buyer_per_maker_base <- t_buy_base / t_maker_base
  t_buyer_per_taker_quote <- t_buy_quote / t_taker_quote
  t_buyer_per_maker_quote <- t_buy_quote / t_maker_quote
    
  for (sequence_length_ in 1 : length(t_lookback_lengths)) {
    table_[glue("ema{t_lookback_lengths[sequence_length_]}")] <- append(
      c(rep(NULL, t_lookback_lengths[sequence_length_] - 1)), 
      t_sma_ls[[sequence_length_]]
    )
    table_[glue("sma{t_lookback_lengths[sequence_length_]}")] <- append(
      c(rep(NULL, t_lookback_lengths[sequence_length_] - 1)), 
      t_ema_ls[[sequence_length_]]
    )
    
    table_[glue("rsi_change{t_lookback_lengths[sequence_length_]}")] <- append(
      c(rep(NULL, t_lookback_lengths[sequence_length_] - 1)), 
      t_rsi_change_buy_ls[[sequence_length_]]
    )
    table_[glue("rsi_vol{t_lookback_lengths[sequence_length_]}")] <- append(
      c(rep(NULL, t_lookback_lengths[sequence_length_] - 1)), 
      t_rsi_vol_buy_ls[[sequence_length_]]
    )
    
    if (sequence_length_ > 1) {
      table_[glue("macd{t_lookback_lengths[sequence_length_]}")] <- append(
        c(rep(NULL, t_lookback_lengths[sequence_length_] - 1)), 
        t_macd_ls[[sequence_length_]]
      )
    }
    
    table_[glue("atr_realized{t_lookback_lengths[sequence_length_]}")] <- append(
      c(rep(NULL, t_lookback_lengths[sequence_length_] - 1)), 
      t_atr_ls[[sequence_length_]]
    )
    table_[glue("atr_traversed{t_lookback_lengths[sequence_length_]}")] <- append(
      c(rep(NULL, t_lookback_lengths[sequence_length_] - 1)), 
      t_atr_extreme_ls[[sequence_length_]]
    )
  }
  
  table_[glue("average_change{t_lookback_lengths[sequence_length_]}")] <- t_average_change
  table_[glue("average_order_size{t_lookback_lengths[sequence_length_]}")] <- t_average_order_size
  table_[glue("average_exchange_price{t_lookback_lengths[sequence_length_]}")] <- t_average_exchange_price
  table_[glue("maker_sell_quote{t_lookback_lengths[sequence_length_]}")] <- t_maker_sell_quote
  table_[glue("buy_base{t_lookback_lengths[sequence_length_]}")] <- t_buy_base
  table_[glue("sell_base{t_lookback_lengths[sequence_length_]}")] <- t_sell_base
  table_[glue("buy_sell_ratio{t_lookback_lengths[sequence_length_]}")] <- t_buy_sell_ratio
  table_[glue("taker_per_maker_base{t_lookback_lengths[sequence_length_]}")] <- t_taker_per_maker_base
  table_[glue("taker_per_maker_quote{t_lookback_lengths[sequence_length_]}")] <- t_taker_per_maker_quote
  table_[glue("buyer_per_taker_base{t_lookback_lengths[sequence_length_]}")] <- t_buyer_per_taker_base
  table_[glue("buyer_per_maker_base{t_lookback_lengths[sequence_length_]}")] <- t_buyer_per_maker_base
  table_[glue("buyer_per_taker_quote{t_lookback_lengths[sequence_length_]}")] <- t_buyer_per_taker_quote
  table_[glue("buyer_per_maker_quote{t_lookback_lengths[sequence_length_]}")] <- t_buyer_per_maker_quote
  
  return (table_)
}
# TD: run for only 1 ccy >> file

setwd("~/Programming/Algorithmic trading")
feature_timeseries <- read.table(
  "broadened_feature_timeseries2.csv", sep = ",", dec = ".", quote = "", 
  col.names = c(
    "pair symbol", "open timestamp", "open", "high", "low", "close", "volume", "close timestamp",
    "quote volume", "trade quantity", "taker buy base", "taker buy quote"
  ),
  nrows = -1 # Modify to number of rows if only want to load some currencies for testing, default -1 = all
)
# TD: append colnames to csv

pairs_list <- unique(feature_timeseries$pair.symbol)

feature_timeseries_filtered_ls <- cleanData(feature_timeseries, pairs_list)

for (table_ in feature_timeseries_filtered_ls){
  write.csv(appendCalculatedMetrics(table_), "expanded_feature_timeseries.csv")
}