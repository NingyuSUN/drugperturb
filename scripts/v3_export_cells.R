# Cell-level frozen-gene counts for one cell type; never computes treatment effects.
# Rscript scripts/v3_export_cells.R INPUT_RDS WELL_MAPPING GENE_MANIFEST OUTPUT_DIR CELL_TYPE TIME_HOURS
# Writes cells.tsv.gz, genes.tsv, CSR int32 arrays (cells x frozen genes) and a summary.
source("scripts/export_context_rds.R",local=TRUE)

export_cells <- function(source,mapping,gene_manifest,cell_type,time_hours) {
  csc <- source$csc; cells <- source$cells; cell_ids <- source$cell_ids
  validate_context_csc(csc)
  wells <- deduplicate_context_wells(mapping,cell_type,time_hours)
  gene_ids <- as.character(source$genes$gene_id)
  frozen <- as.character(gene_manifest$gene_id)
  if (anyDuplicated(frozen) || !all(frozen %in% gene_ids)) stop("Frozen genes absent or duplicated in source")
  if (!identical(as.integer(gene_manifest$output_index),seq_along(frozen)-1L)) stop("Frozen gene order is not 0..n-1")
  qc_fields <- c("hash_umis_W","hash_umis_P","top_to_second_best_ratio_W","top_to_second_best_ratio_P","rt_well")
  require_columns(cells,c(context_source_fields,"culture_plate",qc_fields),"Source cell metadata")
  for (field in c("time_point","dose",qc_fields)) {
    cells[[field]] <- suppressWarnings(as.numeric(as.character(cells[[field]])))
    if (anyNA(cells[[field]])) stop("Invalid source numeric metadata")
  }
  if (!all(as.character(cells$cell_type)==cell_type) || !all(cells$time_point==time_hours)) stop("Source context mismatch")
  if (!all(cells$Combo==paste0(cells$well_oligo,cells$plate_oligo))) stop("Source Combo mismatch")
  source_qc <- cells$hash_umis_W>=5 & cells$hash_umis_P>=5 &
    cells$top_to_second_best_ratio_W>=5 & cells$top_to_second_best_ratio_P>=5 & cells$rt_well!=1
  if (anyNA(source_qc) || !all(source_qc)) stop("Source cells fail documented author hash QC")
  positions <- match(as.character(cells$Combo),wells$Combo)
  mapped <- which(!is.na(positions))
  for (field in context_source_fields) {
    if (!identical(as.character(cells[[field]][mapped]),as.character(wells[[field]][positions[mapped]]))) {
      stop(paste("Selected-well identity mismatch:",field))
    }
  }
  # Zero-library cells cannot be library-normalized; they are counted, not exported.
  column_umi <- vapply(mapped,function(j) if (csc$p[j+1]>csc$p[j]) sum(csc$x[seq.int(csc$p[j]+1,csc$p[j+1])]) else 0,numeric(1))
  selected <- mapped[column_umi>0]
  gene_position <- match(gene_ids,frozen)-1L
  indptr <- integer(length(selected)+1L); chunks_i <- list(); chunks_x <- list()
  library_size <- numeric(length(selected)); frozen_umi <- numeric(length(selected))
  offset <- 0L
  for (k in seq_along(selected)) {
    j <- selected[k]; start <- csc$p[j]+1; end <- csc$p[j+1]
    if (end>=start) {
      idx <- seq.int(start,end); rows <- csc$i[idx]+1L; values <- csc$x[idx]
      library_size[k] <- sum(values)
      pos <- gene_position[rows]; keep <- !is.na(pos)
      ord <- order(pos[keep])
      chunks_i[[k]] <- as.integer(pos[keep][ord]); chunks_x[[k]] <- as.integer(values[keep][ord])
      frozen_umi[k] <- sum(values[keep])
      offset <- offset+sum(keep)
    } else {
      chunks_i[[k]] <- integer(0); chunks_x[[k]] <- integer(0)
    }
    indptr[k+1L] <- offset
  }
  indices <- unlist(chunks_i,use.names=FALSE); data <- unlist(chunks_x,use.names=FALSE)
  if (length(indices)!=offset || sum(as.numeric(data))!=sum(frozen_umi)) stop("Frozen-gene UMI conservation failed")
  if (any(library_size<=0)) stop("Selected cell with zero all-gene library")
  keep_cols <- c("Combo","well_oligo","plate_oligo","cell_type","replicate","time_point","treatment","dose","dose_pattern")
  table <- data.frame(cell_id=as.character(cell_ids)[selected],cells[selected,keep_cols,drop=FALSE],
                      human_UMI=library_size,frozen_gene_UMI=frozen_umi,check.names=FALSE,row.names=NULL)
  summary <- data.frame(key=c("cell_type","time_hours","source_nuclei","mapped_nuclei","zero_library_nuclei_excluded",
                              "exported_nuclei","selected_wells_in_mapping","observed_exported_wells","frozen_genes",
                              "stored_entries","frozen_gene_UMI","all_gene_UMI","treatment_effects_computed"),
                        value=c(cell_type,time_hours,csc$Dim[2],length(mapped),length(mapped)-length(selected),
                                length(selected),nrow(wells),length(unique(table$Combo)),length(frozen),
                                length(indices),sum(frozen_umi),sum(library_size),FALSE),stringsAsFactors=FALSE)
  list(table=table,indptr=indptr,indices=indices,data=data,genes=gene_manifest,summary=summary)
}

write_cells <- function(result,output) {
  files <- c("cells.tsv.gz","genes.tsv","indptr.i32","indices.i32","data.i32","summary.tsv")
  if (any(file.exists(file.path(output,files)))) stop("Refusing to overwrite prior cell export")
  dir.create(output,recursive=TRUE,showWarnings=FALSE)
  write_gzip_table(result$table,file.path(output,files[1]))
  write.table(result$genes,file.path(output,files[2]),sep="\t",row.names=FALSE,quote=FALSE)
  for (name in c("indptr","indices","data")) {
    connection <- file(file.path(output,paste0(name,".i32")),"wb")
    writeBin(as.integer(result[[name]]),connection,size=4L,endian="little"); close(connection)
  }
  write.table(result$summary,file.path(output,files[6]),sep="\t",row.names=FALSE,quote=FALSE)
  invisible(TRUE)
}

main_cells <- function(args=commandArgs(trailingOnly=TRUE)) {
  if (length(args)!=6L) stop("Expected INPUT_RDS WELL_MAPPING GENE_MANIFEST OUTPUT_DIR CELL_TYPE TIME_HOURS")
  time <- as.numeric(args[6])
  if (!args[5] %in% c("A549","MCF7","K562") || is.na(time)) stop("Unsupported context")
  mapping <- read.delim(args[2],check.names=FALSE,stringsAsFactors=FALSE,na.strings=NULL)
  genes <- read.delim(args[3],check.names=FALSE,stringsAsFactors=FALSE)
  source <- read_context_source(args[1]); invisible(gc(verbose=FALSE))
  result <- export_cells(source,mapping,genes,args[5],time)
  rm(source); invisible(gc(verbose=FALSE))
  write_cells(result,args[4])
  print(result$summary,row.names=FALSE)
}

if (sys.nframe()==0L) main_cells()
