# Public-source count extraction only: never compute treatment effects here.
# Run from the repository root:
# Rscript scripts/export_context_rds.R INPUT_RDS WELL_MAPPING OUTPUT_DIR CELL_TYPE TIME_HOURS
source("scripts/csc_utils.R",local=TRUE)

require_columns <- function(value,columns,label) {
  if (!is.data.frame(value) || !all(columns %in% names(value))) stop(paste(label,"schema is incomplete"))
}

context_source_fields <- c("Combo","well_oligo","plate_oligo","cell_type","replicate",
                           "time_point","treatment","dose","dose_pattern")

deduplicate_context_wells <- function(mapping,cell_type,time_hours) {
  fields <- c(context_source_fields,"vehicle","row_kind","SMILES")
  require_columns(mapping,fields,"Selected well metadata")
  time <- suppressWarnings(as.numeric(as.character(mapping$time_point)))
  if (anyNA(time) || any(!is.finite(time))) stop("Invalid selected time metadata")
  selected <- mapping[as.character(mapping$cell_type)==cell_type & time==time_hours,fields,drop=FALSE]
  if (!nrow(selected)) stop("No selected wells for requested context")
  for (key in setdiff(fields,c("vehicle","dose","SMILES"))) {
    if (anyNA(selected[[key]]) || any(as.character(selected[[key]])=="")) stop("Missing selected source field")
    selected[[key]] <- as.character(selected[[key]])
  }
  selected$time_point <- as.numeric(selected$time_point)
  selected$dose <- suppressWarnings(as.numeric(as.character(selected$dose)))
  if (anyNA(selected$dose) || any(!is.finite(selected$dose)) || any(selected$dose<0) ||
      any(selected$dose!=floor(selected$dose))) stop("Invalid selected dose")
  vehicle <- tolower(as.character(selected$vehicle))
  if (anyNA(vehicle) || !all(vehicle %in% c("true","false"))) stop("Invalid selected vehicle flag")
  selected$vehicle <- vehicle=="true"
  if (any(selected$vehicle & selected$dose!=0) || any(!selected$vehicle & selected$dose<=0) ||
      any(selected$row_kind!=ifelse(selected$vehicle,"vehicle","drug"))) stop("Inconsistent selected vehicle/dose metadata")
  if (!all(selected$replicate %in% c("rep1","rep2"))) stop("Invalid selected replicate")
  if (!all(selected$Combo==paste0(selected$well_oligo,selected$plate_oligo))) stop("Selected Combo mismatch")
  selected$SMILES <- as.character(selected$SMILES)
  if (anyNA(selected$SMILES)) stop("Missing selected SMILES field")
  for (indexes in split(seq_len(nrow(selected)),selected$Combo)) {
    if (any(vapply(selected[indexes,,drop=FALSE],function(x) length(unique(x))!=1L,logical(1)))) {
      stop("Conflicting source fields for shared Combo")
    }
  }
  selected <- selected[!duplicated(selected$Combo),,drop=FALSE]
  selected <- selected[order(selected$Combo),,drop=FALSE]
  rownames(selected) <- NULL
  selected
}

validate_context_csc <- function(csc) {
  if (!is.list(csc) || !all(c("i","p","x","Dim") %in% names(csc))) stop("Invalid CSC schema")
  for (field in c("i","p","Dim")) {
    values <- csc[[field]]
    if (!is.numeric(values) || anyNA(values) || any(!is.finite(values)) ||
        any(values!=floor(values))) stop("Invalid integer CSC indices or pointers")
  }
  if (!is.numeric(csc$x)) stop("Invalid numeric CSC values")
  validate_csc(csc)
  for (j in seq_len(csc$Dim[2])) {
    start <- csc$p[j]+1; end <- csc$p[j+1]
    if (end>start && any(diff(csc$i[seq.int(start,end)])<=0)) stop("Duplicate or unsorted CSC rows")
  }
  if (!is.finite(sum(csc$x)) || sum(csc$x)>=2^53) stop("Raw UMI totals exceed exact integer range")
  invisible(TRUE)
}

verify_assay_names <- function(csc,gene_ids,cell_ids) {
  names <- csc$dimnames
  if (!is.null(names) && (!is.list(names) || length(names)!=2L)) stop("Invalid assay dimension names")
  gene_present <- !is.null(names) && !is.null(names[[1]])
  cell_present <- !is.null(names) && !is.null(names[[2]])
  if (gene_present && !identical(as.character(names[[1]]),as.character(gene_ids))) stop("Assay gene order differs from rowRanges IDs")
  if (cell_present && !identical(as.character(names[[2]]),as.character(cell_ids))) stop("Assay cell order differs from colData IDs")
  list(counts_gene_dimnames_present=gene_present,counts_cell_dimnames_present=cell_present,
       row_positional_binding_scope=if (gene_present) "Exact assay Dimnames and rowRanges gene_id order" else "Known SummarizedExperiment counts/rowRanges positional schema; assay row names absent",
       col_positional_binding_scope=if (cell_present) "Exact assay Dimnames and colData cell_id order" else "Known SummarizedExperiment counts/colData positional schema; assay column names absent")
}

unpack_context_object <- function(obj) {
  if (!identical(as.character(class(obj)),"cell_data_set")) stop("Unsupported source object class")
  assay_environment <- attr(attr(obj,"assays"),".xData")
  if (!is.environment(assay_environment) || !exists("data",envir=assay_environment,inherits=FALSE)) stop("Unsupported source assay schema")
  assays <- attr(get("data",envir=assay_environment,inherits=FALSE),"listData")
  if (!"counts" %in% names(assays)) stop("Source counts assay absent")
  matrix <- assays[["counts"]]
  if (!identical(as.character(class(matrix)),"dgCMatrix")) stop("Unexpected source counts storage")
  csc <- list(i=attr(matrix,"i"),p=attr(matrix,"p"),x=attr(matrix,"x"),
              Dim=attr(matrix,"Dim"),dimnames=attr(matrix,"Dimnames"))
  cells <- as.data.frame(attr(attr(obj,"colData"),"listData"),stringsAsFactors=FALSE)
  cell_ids <- attr(attr(obj,"colData"),"rownames")
  metadata <- attr(attr(attr(obj,"rowRanges"),"elementMetadata"),"listData")
  if (!all(c("id","gene_short_name") %in% names(metadata))) stop("Source gene metadata absent")
  genes <- data.frame(gene_id=as.character(metadata$id),gene_symbol=as.character(metadata$gene_short_name),stringsAsFactors=FALSE)
  verify_assay_names(csc,genes$gene_id,cell_ids)
  list(csc=csc,cells=cells,genes=genes,cell_ids=cell_ids)
}

read_context_source <- function(path) {
  # Official GEO files have two gzip layers; detect bytes rather than extensions.
  # Plain/single-layer serialized fixtures are supported without guessing format.
  prefix <- function(connection) tryCatch(readBin(connection,"raw",n=2L),finally=close(connection))
  magic <- as.raw(c(0x1f,0x8b))
  outer_gzip <- identical(prefix(file(path,open="rb")),magic)
  layers <- 0L
  if (outer_gzip) {
    inner_gzip <- identical(prefix(gzfile(path,open="rb")),magic)
    layers <- if (inner_gzip) 2L else 1L
    connection <- gzfile(path,open="rb")
    if (inner_gzip) connection <- gzcon(connection)
  } else {
    connection <- file(path,open="rb")
  }
  result <- tryCatch(unpack_context_object(readRDS(connection)),finally=close(connection))
  result$compression_layers <- layers
  result
}

extract_context_counts <- function(csc,cells,genes,cell_ids,mapping,cell_type,time_hours) {
  wells <- deduplicate_context_wells(mapping,cell_type,time_hours)
  validate_context_csc(csc)
  require_columns(genes,c("gene_id","gene_symbol"),"Source genes")
  gene_ids <- as.character(genes$gene_id)
  if (length(gene_ids)!=csc$Dim[1] || anyNA(gene_ids) || anyDuplicated(gene_ids) ||
      !all(grepl("^ENSG[0-9]+(\\.[0-9]+)?(_PAR_Y)?$",gene_ids))) stop("Expected unique human ENSG gene rows")
  if (length(cell_ids)!=csc$Dim[2] || anyNA(cell_ids) || anyDuplicated(cell_ids) ||
      any(as.character(cell_ids)=="") || nrow(cells)!=csc$Dim[2]) stop("Source cell identity/dimensions invalid")
  binding <- verify_assay_names(csc,gene_ids,cell_ids)
  qc_fields <- c("hash_umis_W","hash_umis_P","top_to_second_best_ratio_W","top_to_second_best_ratio_P","rt_well")
  require_columns(cells,c(context_source_fields,"culture_plate",qc_fields),"Source cell metadata")
  for (field in c("time_point","dose",qc_fields)) {
    cells[[field]] <- suppressWarnings(as.numeric(as.character(cells[[field]])))
    if (anyNA(cells[[field]])) stop("Invalid source numeric metadata")
  }
  if (any(!is.finite(cells$time_point)) || any(!is.finite(cells$dose)) ||
      !all(as.character(cells$cell_type)==cell_type) || !all(cells$time_point==time_hours)) stop("Source cell/time context mismatch")
  if (anyNA(cells$Combo) || !all(cells$Combo==paste0(cells$well_oligo,cells$plate_oligo)) ||
      !all(as.character(cells$culture_plate)==as.character(cells$plate_oligo))) stop("Source Combo/culture plate mismatch")
  source_qc <- cells$hash_umis_W>=5 & cells$hash_umis_P>=5 &
    cells$top_to_second_best_ratio_W>=5 & cells$top_to_second_best_ratio_P>=5 & cells$rt_well!=1
  if (anyNA(source_qc) || !all(source_qc)) stop("Source cells fail documented author hash QC")
  positions <- match(as.character(cells$Combo),wells$Combo)
  selected <- which(!is.na(positions))
  for (field in context_source_fields) {
    expected <- wells[[field]][positions[selected]]
    observed <- cells[[field]][selected]
    if (anyNA(observed) || !identical(as.character(observed),as.character(expected))) {
      stop(paste("Source selected-well identity mismatch:",field))
    }
  }
  result <- aggregate_csc(csc,as.character(cells$Combo),wells$Combo)
  rownames(result$counts) <- gene_ids
  kept <- which(!is.na(result$library_sizes))
  if (!identical(kept,selected) || sum(result$counts)!=sum(result$library_sizes[kept])) stop("Selected source UMI conservation failed")
  cell_qc <- data.frame(cell_id=as.character(cell_ids)[kept],cells[kept,c(context_source_fields,qc_fields),drop=FALSE],
                       human_UMI=result$library_sizes[kept],human_detected_genes=result$n_genes[kept],check.names=FALSE)
  well_qc <- data.frame(Combo=wells$Combo,cell_type=cell_type,time_point=time_hours,
                       n_cells=as.integer(result$n_cells),human_UMI=colSums(result$counts),
                       genes_detected=colSums(result$counts>0),source_observed=result$n_cells>0,
                       check.names=FALSE,row.names=NULL)
  fields <- c(list(source_human_genes=length(gene_ids),source_nuclei=csc$Dim[2],
    source_stored_count_entries=length(csc$x),source_total_UMI=sum(csc$x),
    context_cell_type=cell_type,context_time_hours=time_hours,selected_nuclei=length(kept),
    selected_unique_wells=nrow(wells),nonempty_selected_wells=sum(result$n_cells>0),
    missing_selected_wells=sum(result$n_cells==0),selected_UMI=sum(result$counts),
    zero_library_selected_cells=sum(result$library_sizes[kept]==0),
    counts_nonnegative_integer=TRUE,source_hash_QC_all_pass=TRUE,UMI_conserved=TRUE,
    all_human_gene_rows_exported=TRUE,author_embeddings_reused=FALSE,treatment_effects_computed=FALSE),binding)
  summary <- data.frame(key=names(fields),value=vapply(fields,as.character,character(1)),stringsAsFactors=FALSE)
  list(counts=result$counts,gene_metadata=genes,well_qc=well_qc,selected_cell_qc=cell_qc,
       extraction_summary=summary)
}

write_context_outputs <- function(result,output) {
  files <- c("pseudobulk_counts.tsv.gz","gene_metadata.tsv","well_qc.tsv",
             "selected_cell_qc.tsv.gz","extraction_summary.tsv")
  if (any(file.exists(file.path(output,files)))) stop("Refusing to overwrite prior context extraction outputs")
  if (!nzchar(Sys.which("gzip"))) stop("gzip integrity checker is required")
  dir.create(output,recursive=TRUE,showWarnings=FALSE)
  staged <- tempfile(pattern=".context-export-",tmpdir=output)
  dir.create(staged)
  on.exit(unlink(staged,recursive=TRUE),add=TRUE)
  write_gzip_table(data.frame(gene_id=result$gene_metadata$gene_id,result$counts,check.names=FALSE),
                   file.path(staged,files[1]))
  write.table(result$gene_metadata,file.path(staged,files[2]),sep="\t",row.names=FALSE,quote=FALSE)
  write.table(result$well_qc,file.path(staged,files[3]),sep="\t",row.names=FALSE,quote=FALSE)
  write_gzip_table(result$selected_cell_qc,file.path(staged,files[4]))
  write.table(result$extraction_summary,file.path(staged,files[5]),sep="\t",row.names=FALSE,quote=FALSE)
  for (name in files[c(1,4)]) {
    if (system2("gzip",c("-t",shQuote(file.path(staged,name))))!=0L) stop("Completed gzip integrity check failed")
  }
  # The summary is the last publication marker; consumers must require it.
  for (name in files) if (!file.rename(file.path(staged,name),file.path(output,name))) stop("Atomic output publication failed")
  invisible(TRUE)
}

main <- function(args=commandArgs(trailingOnly=TRUE)) {
  if (length(args)!=5L) stop("Expected INPUT_RDS WELL_MAPPING OUTPUT_DIR CELL_TYPE TIME_HOURS")
  time <- suppressWarnings(as.numeric(args[5]))
  if (length(time)!=1L || is.na(time) || !is.finite(time) || time<=0 || time!=floor(time)) stop("Invalid requested time")
  if (!args[4] %in% c("A549","MCF7","K562")) stop("Unsupported requested cell type")
  mapping <- read.delim(args[2],check.names=FALSE,stringsAsFactors=FALSE,na.strings=NULL)
  source <- read_context_source(args[1])
  invisible(gc(verbose=FALSE))
  result <- extract_context_counts(source$csc,source$cells,source$genes,source$cell_ids,mapping,args[4],time)
  result$extraction_summary <- rbind(result$extraction_summary,
    data.frame(key="source_compression_layers",value=as.character(source$compression_layers),stringsAsFactors=FALSE))
  write_context_outputs(result,args[3])
  print(result$extraction_summary,row.names=FALSE)
  invisible(result$extraction_summary)
}

if (sys.nframe()==0L) main()
