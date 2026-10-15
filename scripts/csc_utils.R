# Aggregate validated zero-based CSC arrays without loading legacy S4 methods.
validate_csc <- function(csc) {
 stopifnot(length(csc$Dim)==2L,all(csc$Dim>=1),length(csc$p)==csc$Dim[2]+1L)
 if (length(csc$i)!=length(csc$x) || csc$p[1]!=0 || tail(csc$p,1)!=length(csc$x) || any(diff(csc$p)<0)) stop("Invalid CSC pointers")
 if (any(!is.finite(csc$x)) || any(csc$x<0) || any(csc$x!=floor(csc$x))) stop("Invalid raw UMI values")
 if (any(csc$i<0) || any(csc$i>=csc$Dim[1]) || any(csc$i!=floor(csc$i))) stop("Invalid CSC row indices")
 invisible(TRUE)
}
aggregate_csc <- function(csc,cell_wells,selected_wells) {
 validate_csc(csc)
 if (length(cell_wells)!=csc$Dim[2] || anyNA(cell_wells)) stop("Invalid cell mapping")
 if (anyDuplicated(selected_wells) || anyNA(selected_wells)) stop("Duplicate selected wells")
 counts <- matrix(0,nrow=csc$Dim[1],ncol=length(selected_wells),dimnames=list(NULL,selected_wells))
 group <- match(cell_wells,selected_wells)
 n_cells <- setNames(tabulate(group,nbins=length(selected_wells)),selected_wells)
 lib <- rep(NA_real_,length(cell_wells)); detected <- rep(NA_integer_,length(cell_wells))
 for (j in which(!is.na(group))) {
  start <- csc$p[j]+1L; end <- csc$p[j+1L]
  if (end<start) {lib[j]<-0;detected[j]<-0L;next}
  indexes <- seq.int(start,end); rows <- csc$i[indexes]+1L; values <- csc$x[indexes]
  if (length(rows)>1L && any(diff(rows)<=0)) stop("Duplicate or unsorted row indices inside column")
  counts[rows,group[j]] <- counts[rows,group[j]]+values
  lib[j] <- sum(values);detected[j] <- sum(values>0)
 }
 if (sum(counts)!=sum(lib,na.rm=TRUE)) stop("Source UMI conservation failed")
 list(counts=counts,n_cells=n_cells,library_sizes=lib,n_genes=detected)
}

# Explicitly close gzip streams before publication: process exit is not a flush.
write_gzip_table <- function(value,path) {
  temporary <- paste0(path,".part")
  connection <- gzfile(temporary,"wt")
  tryCatch(write.table(value,connection,sep="\t",row.names=FALSE,quote=FALSE),
           finally=close(connection))
  if (!file.rename(temporary,path)) stop("Could not publish completed gzip file")
}
