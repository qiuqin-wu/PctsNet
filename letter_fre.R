seq <- list()
con <- file("/fs/scratch/PCON0022/wangqi/WQQ/CCAN/data/cell_line_scGNN/peak_seq2.txt",open="r")
n=1
while ( TRUE ) {
  line = readLines(con, n = 1)
  
  if ( length(line) == 0 ) {
    break
  }
  seq <- c(seq,line)
  n=n+1
}
close(con)

seq2 <- lapply(1:length(seq),function(x){
  line <- unlist(strsplit(seq[[x]],split=''))
  line[line=="A" | line=="a"] <- 0
  line[line=="T" | line=="t"] <- 1
  line[line=="C" | line=="c"] <- 2
  line[line=="G" | line=="g"] <- 3
  line2 <- line
  line3 <- as.numeric(line2)
})



seq <- unlist(seq2)
E_pro_A <- length(which(seq==0))/length(seq)
E_pro_T <- length(which(seq==1))/length(seq)
E_pro_C <- length(which(seq==2))/length(seq)
E_pro_G <- length(which(seq==3))/length(seq)
cat("E_shijigailv:","E_pro_A:",E_pro_A,"E_pro_T:",E_pro_T,"E_pro_C:",E_pro_C,"E_pro_G:",E_pro_G,"\n")


