counts <- readRDS("G:/CCAN2/data/cell_line_scGNN/cell_line_matrix.RDS")
metadata <- read.table('G:/CCAN2/data/cell_line_scGNN/metadata.tsv',
                       header = TRUE,
                       stringsAsFactors=FALSE,quote="",row.names=1)
rowsum <- rowSums(counts)
colsum <- colSums(counts)
index1 <- which(rowsum < 10)
index2 <- which(colsum < 200)
if(length(index1) > 0){
  counts <- counts[-index1,]
}
if(length(index2) > 0){
  counts <- counts[,-index2]
  metadata_filter <- as.data.frame(metadata[-index2,])
  rownames(metadata_filter) <- rownames(metadata)[-index2]
  colnames(metadata_filter) <- "label"
  write.table(metadata_filter, "G:/CCAN2/data/cell_line_scGNN/metadata_filter.tsv", sep = "\t",row.names =TRUE, col.names = TRUE, quote = FALSE)
  
}else{
  metadata_filter <- metadata
  write.table(metadata_filter, "D:/wqq/CCAN/data/GM12878_HL60/code/metadata_filter.tsv", sep = "\t",row.names =TRUE, col.names = TRUE, quote = FALSE)
} 

rownames(counts) <- sub("-",":",rownames(counts))
saveRDS(counts,"G:/CCAN2/data/cell_line_scGNN/count_matrix_filter.RDS")
write.table(counts, "G:/CCAN2/data/cell_line_scGNN/count_matrix_filter.csv", sep = ",", row.names = TRUE, col.names = TRUE, quote = FALSE)

peak <- rownames(counts)
write(peak, "G:/CCAN2/data/cell_line_scGNN/peak_filter.txt", sep = "\n")


