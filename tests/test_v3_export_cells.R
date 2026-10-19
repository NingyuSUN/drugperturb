# Synthetic engineering checks only. Run from the repository root.
source("scripts/v3_export_cells.R")

expect_error <- function(expr, pattern=NULL) {
  error <- tryCatch({force(expr);NULL}, error=function(e)e)
  stopifnot(inherits(error,"error"))
  if (!is.null(pattern)) stopifnot(grepl(pattern,conditionMessage(error)))
  invisible(TRUE)
}

# Four source cells x three genes; cell 4 belongs to an unmapped well.
fixture <- function() {
  csc <- list(i=as.integer(c(0,2,1,2,0,1)),p=as.integer(c(0,2,3,4,6)),
              x=c(2,3,4,5,6,1),Dim=as.integer(c(3,4)))
  cells <- data.frame(Combo=c("D1p1","C2p1","D1p2","unusedp2"),
    well_oligo=c("D1","C2","D1","unused"),plate_oligo=c("p1","p1","p2","p2"),
    culture_plate=c("p1","p1","p2","p2"),cell_type="MCF7",replicate=c("rep1","rep1","rep2","rep2"),
    time_point=24,treatment=c("d1","vehicle","d1","unused"),dose=c(1000,0,1000,1000),
    dose_pattern="2",hash_umis_W=5,hash_umis_P=5,
    top_to_second_best_ratio_W=5,top_to_second_best_ratio_P=5,rt_well=2,stringsAsFactors=FALSE)
  wells <- cells[1:3,c("Combo","well_oligo","plate_oligo","cell_type","replicate","time_point","treatment","dose","dose_pattern")]
  wells$vehicle <- wells$treatment=="vehicle"
  wells$row_kind <- ifelse(wells$vehicle,"vehicle","drug")
  wells$SMILES <- ifelse(wells$vehicle,"","CCO")
  genes <- data.frame(gene_id=c("ENSG000001.1","ENSG000002.2","ENSG000003.1"),
                      gene_symbol=c("a","b","c"),stringsAsFactors=FALSE)
  # Frozen order differs from source order and omits gene 2.
  frozen <- data.frame(output_index=0:1,gene_id=c("ENSG000003.1","ENSG000001.1"),gene_symbol=c("c","a"))
  list(source=list(csc=csc,cells=cells,genes=genes,cell_ids=paste0("cell",1:4)),wells=wells,frozen=frozen)
}

f <- fixture()
r <- export_cells(f$source,f$wells,f$frozen,"MCF7",24)
stopifnot(identical(r$table$cell_id,c("cell1","cell2","cell3")),
  identical(r$table$human_UMI,c(5,4,5)),          # all-gene libraries, including unexported gene 2
  identical(r$table$frozen_gene_UMI,c(5,0,5)),
  identical(as.integer(r$indptr),c(0L,2L,2L,3L)),
  identical(as.integer(r$indices),c(0L,1L,0L)),   # frozen positions, sorted within each cell
  identical(as.integer(r$data),c(3L,2L,5L)))

bad <- fixture();bad$frozen$gene_id[2] <- "ENSG000099.1";expect_error(export_cells(bad$source,bad$wells,bad$frozen,"MCF7",24),"Frozen")
bad <- fixture();bad$frozen$output_index <- c(1L,0L);expect_error(export_cells(bad$source,bad$wells,bad$frozen,"MCF7",24),"order")
bad <- fixture();bad$source$cells$hash_umis_W[4] <- 4;expect_error(export_cells(bad$source,bad$wells,bad$frozen,"MCF7",24),"hash QC")
bad <- fixture();bad$source$cells$replicate[1] <- "rep2";expect_error(export_cells(bad$source,bad$wells,bad$frozen,"MCF7",24),"identity")
zero <- fixture();zero$source$csc$x[3] <- 0     # cell2 now has an all-zero library
z <- export_cells(zero$source,zero$wells,zero$frozen,"MCF7",24)
stopifnot(identical(z$table$cell_id,c("cell1","cell3")),
  z$summary$value[z$summary$key=="zero_library_nuclei_excluded"]=="1")

output <- tempfile()
write_cells(r,output)
read_i32 <- function(name,n) readBin(file.path(output,name),"integer",n=n,size=4L,endian="little")
stopifnot(identical(read_i32("indptr.i32",4L),c(0L,2L,2L,3L)),identical(read_i32("data.i32",3L),c(3L,2L,5L)))
stopifnot(system2("gzip",c("-t",shQuote(file.path(output,"cells.tsv.gz"))))==0L)
expect_error(write_cells(r,output),"overwrite")
unlink(output,recursive=TRUE)
cat("PASS: v3 cell export frozen-gene order, all-gene libraries, CSR arrays, QC and overwrite refusal\n")
