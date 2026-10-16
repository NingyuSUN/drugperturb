# Synthetic engineering checks only. Run from the repository root.
source("scripts/export_context_rds.R")

expect_error <- function(expr, pattern=NULL) {
  error <- tryCatch({force(expr);NULL}, error=function(e)e)
  stopifnot(inherits(error,"error"))
  if (!is.null(pattern)) stopifnot(grepl(pattern,conditionMessage(error)))
  invisible(TRUE)
}

fixture <- function() {
  csc <- list(i=as.integer(c(0,2,1,2,0)),p=as.integer(c(0,2,3,4,5)),
              x=c(2,3,4,5,6),Dim=as.integer(c(3,4)))
  cells <- data.frame(Combo=c("D1p1","C2p1","D1p2","unusedp2"),
    well_oligo=c("D1","C2","D1","unused"),plate_oligo=c("p1","p1","p2","p2"),
    culture_plate=c("p1","p1","p2","p2"),cell_type="MCF7",replicate=c("rep1","rep1","rep2","rep2"),
    time_point=24,treatment=c("d1","vehicle","d1","unused"),dose=c(1000,0,1000,1000),
    dose_pattern="2",hash_umis_W=5,hash_umis_P=5,
    top_to_second_best_ratio_W=5,top_to_second_best_ratio_P=5,rt_well=2,
    stringsAsFactors=FALSE)
  wells <- cells[1:3,c("Combo","well_oligo","plate_oligo","cell_type","replicate","time_point","treatment","dose","dose_pattern")]
  missing <- wells[2,,drop=FALSE]
  missing$Combo <- "C2p2";missing$plate_oligo <- "p2";missing$replicate <- "rep2"
  wells <- rbind(wells,missing)
  wells$vehicle <- wells$treatment=="vehicle"
  wells$row_kind <- ifelse(wells$vehicle,"vehicle","drug")
  wells$SMILES <- ifelse(wells$vehicle,"","CCO")
  wells$context_id <- "MCF7_24h_1000nM"
  shared <- wells[wells$Combo=="C2p1",,drop=FALSE]
  shared$context_id <- "MCF7_24h_100nM"
  wells <- rbind(wells,shared)
  genes <- data.frame(gene_id=c("ENSG000001.1","ENSG000002.2","ENSG000003.1"),
                      gene_symbol=c("marker","other","background"),stringsAsFactors=FALSE)
  list(csc=csc,cells=cells,genes=genes,cell_ids=paste0("cell",1:4),wells=wells)
}

run_fixture <- function(f=fixture()) {
  extract_context_counts(f$csc,f$cells,f$genes,f$cell_ids,f$wells,"MCF7",24)
}

f <- fixture()
dedup <- deduplicate_context_wells(f$wells,"MCF7",24)
stopifnot(nrow(dedup)==4L,!anyDuplicated(dedup$Combo))
bad <- f$wells;bad$replicate[nrow(bad)] <- "rep2"
expect_error(deduplicate_context_wells(bad,"MCF7",24),"[Cc]onflict")
expect_error(deduplicate_context_wells(f$wells,"K562",24),"selected")

r <- run_fixture()
stopifnot(nrow(r$counts)==3L,ncol(r$counts)==4L,
  identical(unname(r$counts[,"D1p1"]),c(2,0,3)),
  identical(unname(r$counts[,"D1p2"]),c(0,0,5)),
  identical(unname(r$counts[,"C2p1"]),c(0,4,0)),
  all(r$counts[,"C2p2"]==0),sum(r$counts)==14,
  identical(r$gene_metadata$gene_id,f$genes$gene_id),
  r$well_qc$n_cells[r$well_qc$Combo=="C2p2"]==0L,
  !r$well_qc$source_observed[r$well_qc$Combo=="C2p2"],
  r$well_qc$n_cells[r$well_qc$Combo=="C2p1"]==1L,
  nrow(r$selected_cell_qc)==3L,
  r$selected_cell_qc$human_UMI[r$selected_cell_qc$cell_id=="cell1"]==5,
  sum(r$selected_cell_qc$human_UMI)==14)

# Context and metadata identity must fail before aggregation or export.
for (field in c("cell_type","time_point","replicate","treatment","dose","dose_pattern","culture_plate")) {
  bad <- fixture()
  if (field %in% c("dose","time_point")) bad$cells[1,field] <- 72 else bad$cells[1,field] <- "wrong"
  expect_error(run_fixture(bad))
}
for (field in c("hash_umis_W","hash_umis_P","top_to_second_best_ratio_W","top_to_second_best_ratio_P")) {
  bad <- fixture();bad$cells[4,field] <- 4
  expect_error(run_fixture(bad),"hash QC") # Includes unselected source cells.
}
bad <- fixture();bad$cells$rt_well[1] <- 1;expect_error(run_fixture(bad),"hash QC")
bad <- fixture();bad$cell_ids[2] <- bad$cell_ids[1];expect_error(run_fixture(bad),"cell")
bad <- fixture();bad$genes$gene_id[2] <- bad$genes$gene_id[1];expect_error(run_fixture(bad),"gene")
bad <- fixture();bad$genes$gene_id[3] <- "ENSMUSG000001";expect_error(run_fixture(bad),"human|gene")
# GENCODE pseudoautosomal Y copies are real human rows, not identifiers to strip.
par_y <- fixture();par_y$genes$gene_id[3] <- "ENSG000003.1_PAR_Y"
par_result <- run_fixture(par_y)
stopifnot(identical(par_result$gene_metadata$gene_id,par_y$genes$gene_id),
          sum(par_result$counts)==14)
bad <- fixture();bad$cells$Combo[1] <- "invented";expect_error(run_fixture(bad),"Combo")
for (value in c(-1,1.5,Inf,NA_real_)) {
  bad <- fixture();bad$csc$x[1] <- value;expect_error(run_fixture(bad))
}
bad <- fixture();bad$csc$p[2] <- 1.5;expect_error(run_fixture(bad),"CSC")
bad <- fixture();bad$csc$i[2] <- 0L;expect_error(run_fixture(bad),"CSC|row")

# A zero-library cell remains a measured cell; the library denominator uses all genes.
bad <- fixture();bad$csc$x[1:2] <- 0
zero <- run_fixture(bad)
stopifnot(zero$well_qc$n_cells[zero$well_qc$Combo=="D1p1"]==1L,
  zero$selected_cell_qc$human_UMI[zero$selected_cell_qc$cell_id=="cell1"]==0)

# Mock only the serialized slot schema, not any aggregation operation.
make_object <- function(f) {
  matrix <- structure(list(),class="dgCMatrix",i=f$csc$i,p=f$csc$p,x=f$csc$x,Dim=f$csc$Dim,
                      Dimnames=list(f$genes$gene_id,f$cell_ids))
  assay_environment <- new.env(parent=emptyenv())
  assay_environment$data <- structure(list(),listData=list(counts=matrix))
  structure(list(),class="cell_data_set",
    assays=structure(list(),.xData=assay_environment),
    colData=structure(list(),listData=as.list(f$cells),rownames=f$cell_ids),
    rowRanges=structure(list(),elementMetadata=structure(list(),
      listData=list(id=f$genes$gene_id,gene_short_name=f$genes$gene_symbol))))
}
obj <- make_object(f)
unpacked <- unpack_context_object(obj)
stopifnot(identical(unpacked$genes$gene_id,f$genes$gene_id))
wrong <- fixture();wrong$genes$gene_id <- rev(wrong$genes$gene_id)
obj_wrong <- obj
attr(attr(attr(obj_wrong,"rowRanges"),"elementMetadata"),"listData")$id <- wrong$genes$gene_id
expect_error(unpack_context_object(obj_wrong),"gene order")

# GEO supplies gzip-compressed RDS bytes wrapped in an additional gzip stream.
# Test plain, single and double layers with a tiny serialized object, not RNA data.
plain <- tempfile();single <- tempfile(fileext=".RDS.gz");double <- tempfile(fileext=".RDS.gz")
saveRDS(obj,plain,compress=FALSE)
saveRDS(obj,single,compress="gzip")
stopifnot(system2("gzip",c("-c",shQuote(single)),stdout=double)==0L)
for (item in list(list(path=plain,layers=0L),list(path=single,layers=1L),list(path=double,layers=2L))) {
  decoded <- read_context_source(item$path)
  stopifnot(decoded$compression_layers==item$layers,
            identical(decoded$genes$gene_id,f$genes$gene_id),
            identical(decoded$csc$x,f$csc$x))
}
unlink(c(plain,single,double))

# Publication verifies closed gzip trailers and refuses overwrite of prior outputs.
output <- tempfile();dir.create(output)
write_context_outputs(r,output)
stopifnot(identical(sort(list.files(output)),sort(c("pseudobulk_counts.tsv.gz","selected_cell_qc.tsv.gz",
            "gene_metadata.tsv","well_qc.tsv","extraction_summary.tsv"))))
for (name in c("pseudobulk_counts.tsv.gz","selected_cell_qc.tsv.gz")) {
  stopifnot(system2("gzip",c("-t",shQuote(file.path(output,name))))==0L)
}
counts <- read.delim(gzfile(file.path(output,"pseudobulk_counts.tsv.gz")),check.names=FALSE)
stopifnot(sum(counts[,-1])==14,nrow(counts)==3L)
expect_error(write_context_outputs(r,output),"overwrite")
unlink(output,recursive=TRUE)
cat("PASS: context/well dedup, source identity, missing wells, human genes, CSC/count QC, UMI conservation, gzip atomic publication\n")
