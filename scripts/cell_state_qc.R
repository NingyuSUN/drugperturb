# Descriptive fixed-marker expression; not cell phase or pathway activity.
# Usage: Rscript scripts/cell_state_qc.R RDS CELL_QC MARKER_TSV WELL_QC OUTPUT
# MARKER_TSV: set_id, gene_id; generated exactly from frozen biology_v1.json.
source("scripts/csc_utils.R")

score_marker_cells <- function(csc, selected_columns, set_rows) {
  validate_csc(csc)
  if (anyNA(selected_columns) || anyDuplicated(selected_columns) ||
      any(selected_columns < 1 | selected_columns > csc$Dim[2])) stop("Invalid cell indexes")
  if (!length(set_rows) || any(vapply(set_rows, function(x) !length(x) || anyNA(x) ||
      anyDuplicated(x) || any(x < 1 | x > csc$Dim[1]), logical(1)))) stop("Invalid marker rows")
  membership <- lapply(set_rows, function(rows) seq_len(csc$Dim[1]) %in% rows)
  n <- length(selected_columns)
  library <- numeric(n)
  marker_umi <- matrix(0, nrow=n, ncol=length(set_rows), dimnames=list(NULL,names(set_rows)))
  for (k in seq_along(selected_columns)) {
    j <- selected_columns[k]
    start <- csc$p[j]+1L; end <- csc$p[j+1L]
    if (end < start) next
    indexes <- seq.int(start,end)
    rows <- csc$i[indexes]+1L; values <- csc$x[indexes]
    if (length(rows)>1L && any(diff(rows)<=0)) stop("Duplicate or unsorted CSC rows")
    library[k] <- sum(values) # All human genes, including genes outside either set.
    for (s in seq_along(set_rows)) marker_umi[k,s] <- sum(values[membership[[s]][rows]])
  }
  score <- matrix(NA_real_, nrow=n, ncol=length(set_rows),dimnames=dimnames(marker_umi))
  positive <- library > 0
  score[positive,] <- log1p(10000 * marker_umi[positive,,drop=FALSE] / library[positive])
  list(library=library, marker_umi=marker_umi, score=score)
}

# Known raw counts: cells (2,3,5), (0,2,8), and an empty library.
# The background gene changes the denominator; marker-only normalization is wrong.
selftest <- function() {
  toy <- list(i=c(0L,1L,2L,1L,2L),p=c(0L,3L,5L,5L),x=c(2,3,5,2,8),Dim=c(3L,3L))
  x <- score_marker_cells(toy,1:3,list(S=1L,G2M=2L))
  stopifnot(identical(x$library,c(10,10,0)),
    isTRUE(all.equal(unname(x$marker_umi),matrix(c(2,0,0,3,2,0),nrow=3))),
    isTRUE(all.equal(unname(x$score[1:2,]),matrix(c(log1p(2000),0,log1p(3000),log1p(2000)),nrow=2))),
    all(is.na(x$score[3,])),sum(x$library)==20,
    isTRUE(all.equal(unname(quantile(x$score[1:2,1],c(.25,.5,.75))),log1p(2000)*c(.25,.5,.75))))
  bad <- toy;bad$x[1] <- -1
  stopifnot(inherits(try(score_marker_cells(bad,1:3,list(S=1L)),silent=TRUE),"try-error"))
  invisible(TRUE)
}
selftest()
args <- commandArgs(trailingOnly=TRUE)
if (length(args)==1L && args[1]=="--self-test") {
  cat("PASS: known marker counts, all-human denominator, zero library, quartiles, invalid counts\n")
} else {
  if (length(args)!=5L) stop("Expected RDS CELL_QC MARKER_TSV WELL_QC OUTPUT")
  qc <- read.delim(args[2],check.names=FALSE,stringsAsFactors=FALSE)
  sets <- read.delim(args[3],check.names=FALSE,stringsAsFactors=FALSE)
  well_qc <- read.delim(args[4],check.names=FALSE,stringsAsFactors=FALSE)
  stopifnot(nrow(qc)==36848L,!anyDuplicated(qc$cell_id),nrow(well_qc)==392L,
    !anyDuplicated(well_qc$Combo),!anyNA(qc$human_UMI),
    all(is.finite(qc$human_UMI)),all(qc$human_UMI>=0),all(qc$human_UMI==floor(qc$human_UMI)),
    setequal(unique(sets$set_id),c("author_s_phase","author_g2m_phase")),
    !anyDuplicated(sets[c("set_id","gene_id")]))
  con <- gzcon(gzfile(args[1],open="rb"))
  obj <- readRDS(con);close(con)
  stopifnot(identical(as.character(class(obj)),"cell_data_set"))
  ae <- attr(attr(obj,"assays"),".xData")
  assays <- attr(get("data",envir=ae,inherits=FALSE),"listData")
  C <- assays[["counts"]]
  stopifnot(identical(as.character(class(C)),"dgCMatrix"))
  csc <- list(i=attr(C,"i"),p=attr(C,"p"),x=attr(C,"x"),Dim=attr(C,"Dim"))
  cd <- as.data.frame(attr(attr(obj,"colData"),"listData"),stringsAsFactors=FALSE)
  cell_ids <- attr(attr(obj,"colData"),"rownames")
  genes <- attr(attr(attr(obj,"rowRanges"),"elementMetadata"),"listData")
  gene_ids <- as.character(genes$id)
  stopifnot(length(gene_ids)==58347L,csc$Dim[1]==58347L,length(cell_ids)==csc$Dim[2],
    nrow(cd)==length(cell_ids),!anyDuplicated(cell_ids),!anyDuplicated(gene_ids))
  selected_columns <- match(qc$cell_id,cell_ids)
  stopifnot(!anyNA(selected_columns),identical(as.character(cd$Combo[selected_columns]),as.character(qc$Combo)))
  source_qc <- cd$hash_umis_W[selected_columns]>=5 & cd$hash_umis_P[selected_columns]>=5 &
    cd$top_to_second_best_ratio_W[selected_columns]>=5 & cd$top_to_second_best_ratio_P[selected_columns]>=5 & cd$rt_well[selected_columns]!=1
  stopifnot(!anyNA(source_qc),all(source_qc))
  set_rows <- lapply(split(sets$gene_id,sets$set_id),function(ids) match(ids,gene_ids))
  result <- score_marker_cells(csc,selected_columns,set_rows)
  # Exact cell-by-cell comparison to the already extracted QC; no reaggregation.
  stopifnot(identical(as.numeric(result$library),as.numeric(qc$human_UMI)),
    sum(result$library)==59281450,all(result$marker_umi<=result$library))
  rows <- list(); k <- 0L
  for (s in seq_along(set_rows)) for (well in well_qc$Combo) {
    at <- which(qc$Combo==well)
    expected <- well_qc[match(well,well_qc$Combo),]
    stopifnot(length(at)==expected$n_cells,sum(result$library[at])==expected$human_UMI)
    values <- result$score[at,s];valid <- values[is.finite(values)]
    q <- if(length(valid)) quantile(valid,c(.25,.5,.75),names=FALSE,type=7) else rep(NA_real_,3)
    k <- k+1L
    rows[[k]] <- data.frame(Combo=well,set_id=names(set_rows)[s],n_cells=length(at),
      n_scored=length(valid),n_zero_library=sum(result$library[at]==0),
      marker_gene_count=length(set_rows[[s]]),median=q[2],q25=q[1],q75=q[3],
      marker_raw_UMI=sum(result$marker_umi[at,s]),all_human_raw_UMI=sum(result$library[at]))
  }
  summary <- do.call(rbind,rows)
  stopifnot(nrow(summary)==784L,all(table(summary$Combo)==2L))
  write.table(summary,args[5],sep="\t",row.names=FALSE,quote=FALSE,na="NA")
  # Structural counts only: no condition differences or score values are printed.
  cat("PASS: cells=",nrow(qc)," wells=",nrow(well_qc)," genes=",length(gene_ids),
      " sets=",length(set_rows)," summary_rows=",nrow(summary)," total_human_UMI=",sum(result$library),
      " zero_library_cells=",sum(result$library==0)," per_cell_library_match=TRUE\n",sep="")
}
