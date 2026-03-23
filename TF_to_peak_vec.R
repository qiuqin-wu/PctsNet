embedding_data <- read.table("/fs/ess/PCON0022/wangqi/WQQ/CCAN/data/cell_line_scGNN/node2vec_embeddings.txt", skip = 1, header = FALSE,row.names=1)

peak_motif_have <- readRDS("/fs/ess/PCON0022/wangqi/WQQ/CCAN/data/cell_line_scGNN/peak_motif_have.RDS")

motif_TF_predict <- readRDS("/fs/ess/PCON0022/wangqi/WQQ/CCAN/data/cell_line_scGNN/motif_TF_predict.RDS")


TF_to_peak_vec <- function(x){
  motifs <- peak_motif_have$motifs[[x]]
  index <- which(names(motif_TF_predict) %in% motifs)
  TF <- unlist(motif_TF_predict[index])
  index_TF <- which(rownames(embedding_data) %in% TF)
  embedding_data_subset <- embedding_data[index_TF,]
  col_sums <- colSums(embedding_data_subset)
  return(col_sums)
  }


library(foreach)
library(doParallel)
cores <- 30 # 指定使用的核心数
cl <- makeCluster(cores)  # 创建一个并行计算集群
registerDoParallel(cl)  # 注册并行计算集群
TF_to_peak_vec2 <- foreach(x=1:nrow(peak_motif_have)) %dopar% TF_to_peak_vec(x)

# 停止并行计算集群
stopCluster(cl)

saveRDS(TF_to_peak_vec2,"/fs/ess/PCON0022/wangqi/WQQ/CCAN/data/cell_line_scGNN/TF_to_peak_vec2.RDS")
