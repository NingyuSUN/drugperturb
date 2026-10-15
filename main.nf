#!/usr/bin/env nextflow
nextflow.enable.dsl=2
import groovy.json.JsonSlurper
import groovy.json.JsonOutput
import java.security.MessageDigest

String quoted(Object text) { "'" + text.toString().replace("'", "'\"'\"'") + "'" }
String digestFile(java.nio.file.Path path) {
    def digest = MessageDigest.getInstance('SHA-256')
    path.toFile().withInputStream { stream ->
        byte[] buffer = new byte[1024*1024]
        int n
        while ((n=stream.read(buffer)) > 0) digest.update(buffer,0,n)
    }
    return digest.digest().encodeHex().toString()
}
void verifySnapshot(java.nio.file.Path root) {
    def receipt = new JsonSlurper().parse(root.resolve('receipts/workflow_run.json').toFile())
    if (receipt.status != 'complete' || !receipt.output_hashes) error('Incomplete workflow receipt')
    def hashes=[:] + receipt.output_hashes + (receipt.input_hashes ?: [:])
    hashes.each { relative, expected ->
        def target=root.resolve(relative).normalize()
        if (!target.startsWith(root) || !java.nio.file.Files.isRegularFile(target))
            error("Missing or unsafe cached output: ${relative}")
        if (digestFile(target) != expected) error("Cached output hash mismatch: ${relative}")
    }
}

process RUN_SNAPSHOT {
    tag "${mode}"
    cache 'deep'
    cpus params.cpus
    memory params.memory
    time params.time_limit
    publishDir params.outdir, mode:'symlink', overwrite:false
    input:
    val mode
    path source_assets
    path input_manifest, stageAs:'input_manifest.json'
    path data_files, stageAs:'inputs/*'
    val settings
    val runtime_identity
    output:
    path 'snapshot', emit:snapshot
    script:
    """
    ${quoted(params.python)} scripts/smoke_pipeline.py \
      --mode ${quoted(mode)} --source-root . --snapshot snapshot \
      --input-manifest input_manifest.json --staged-inputs inputs \
      --settings ${quoted(settings)}
    """
}

workflow {
    if (!params.project_root) error('Pass an absolute --project_root; source files are staged explicitly')
    def root=file(params.project_root).toAbsolutePath().normalize()
    if (!params.project_root.toString().startsWith('/')) error('--project_root must be absolute')
    def mode=params.mode.toString()
    if (!(mode in ['test','core','raw'])) error('--mode must be test, core or raw')
    if (mode != 'test' && !params.allow_real_execution.toString().toBoolean())
        error('Real execution is gated: pass --allow_real_execution only for an intended new isolated rerun')
    if (!params.python) params.python=root.resolve('.venv/bin/python').toString()
    def assets=['src','scripts','configs','locks','conf','containers','docs','main.nf','nextflow.config','pyproject.toml','AGENTS.md','README.md'].collect { name ->
        def asset=root.resolve(name)
        if (!java.nio.file.Files.exists(asset)) error("Missing source asset: ${name}")
        asset
    }
    def manifest=[]
    def data=[]
    if (mode != 'test') {
        if (!params.input_manifest) error('Real execution requires --input_manifest with immutable SHA256 entries')
        manifest=file(params.input_manifest).toAbsolutePath()
        def contents=new JsonSlurper().parse(manifest.toFile())
        if (contents.mode != mode || !contents.artifact_hashes) error('Input manifest mode/hash gate failed')
        def metadata='data/raw/GSM4150378_sciPlex3_A549_MCF7_K562_hashTable_metadata.txt.gz'
        def allowed=mode=='core' ? [metadata,'data/processed/pseudobulk_counts.tsv.gz','data/processed/well_qc.tsv','data/processed/gene_metadata.tsv','data/processed/extraction_summary.tsv','data/processed/selected_cell_qc.tsv.gz','data/processed/cell_state_well_summary.tsv','manifests/source_manifest.tsv'] : [metadata,'data/raw/GSM4150378_sciPlex3_A549_24hrs.RDS.gz','manifests/source_manifest.tsv']
        if (contents.schema_version != 1 || !allowed.containsAll(contents.artifact_hashes.keySet()))
            error('Input manifest contains entries outside exact mode-specific allowlist')
        def names=[]
        contents.artifact_hashes.keySet().sort().each { relative ->
            def candidate=root.resolve(relative).normalize()
            if (relative.startsWith('/') || !candidate.startsWith(root) || !java.nio.file.Files.isRegularFile(candidate))
                error("Missing or unsafe declared input: ${relative}")
            if (candidate.toRealPath().startsWith(root.toRealPath()) == false) error("Input symlink escapes project: ${relative}")
            names.add(candidate.fileName.toString()); data.add(candidate)
        }
        if (names.toSet().size() != names.size()) error('Declared inputs require distinct basenames for safe staging')
    }
    def identity
    if (params.container_image) {
        if (!(params.container_image.toString() ==~ /(?:sha256:[0-9a-f]{64}|.+@sha256:[0-9a-f]{64})/))
            error('Container profile requires an immutable image ID/digest via --container_image')
        identity=params.container_image.toString()
    } else {
        def probeArgs=[params.python.toString(),root.resolve('scripts/smoke_pipeline.py').toString(),'--runtime-json']
        if (mode == 'raw') probeArgs.addAll(['--rscript',params.rscript.toString()])
        def probe=probeArgs.execute()
        probe.waitFor()
        if (probe.exitValue()!=0) error("Python runtime probe failed: ${probe.err.text}")
        identity=probe.in.text.trim()
    }
    def settings=JsonOutput.toJson([source_commit:params.source_commit,rscript:params.rscript,
                                  execution_site:params.execution_site,
                                  container_image:params.container_image,python:params.python])
    RUN_SNAPSHOT(mode,assets,manifest,data,settings,identity)
    // This runs on BOTH fresh and cached tasks; existence alone is insufficient.
    RUN_SNAPSHOT.out.snapshot.map { snapshot ->
        verifySnapshot(snapshot)
        snapshot
    }.view { snapshot -> "Verified snapshot and output hashes: ${snapshot}" }
}


// Linux kernels without /proc/<pid>/io cannot use Nextflow25.10.2's metrics
// wrapper. Explicit portable mode keeps the DAG and honest execution metadata.
workflow.onComplete {
    if (!params.resource_metrics.toString().toBoolean()) {
        def output=file(params.outdir)
        java.nio.file.Files.createDirectories(output)
        def stats=workflow.stats
        def state=workflow.success ? (stats.cachedCount ? 'CACHED' : 'COMPLETED') : 'FAILED'
        def record=[status:state,success:workflow.success,run_name:workflow.runName,
                    session_id:workflow.sessionId.toString(),duration:workflow.duration.toString(),
                    completed_tasks:stats.succeededCount,cached_tasks:stats.cachedCount,
                    failed_tasks:stats.failedCount,resource_metrics:'UNAVAILABLE explicitly disabled',
                    reason:'Portable mode: kernel may lack /proc/PID/io; no CPU/memory/IO values claimed',
                    command:workflow.commandLine]
        def trace=output.resolve('trace.tsv')
        def report=output.resolve('execution_report.html')
        def receipt=output.resolve('workflow_execution.json')
        def timeline=output.resolve('timeline.html')
        if ([trace,report,receipt,timeline].any { java.nio.file.Files.exists(it) })
            throw new IllegalStateException('Preserve existing operational reports; use a fresh --outdir')
        trace.toFile().text='status\tduration\tcompleted_tasks\tcached_tasks\tfailed_tasks\tresource_metrics\n'+
            [state,record.duration,record.completed_tasks,record.cached_tasks,record.failed_tasks,'UNAVAILABLE'].join('\t')+'\n'
        receipt.toFile().text=JsonOutput.prettyPrint(JsonOutput.toJson(record))+'\n'
        timeline.toFile().text='<!doctype html><meta charset="utf-8"><h1>Operational single-process timeline</h1><p>Explicit portable monitoring; not the standard Nextflow resource timeline.</p><p>Workflow state: '+state+'; elapsed: '+record.duration+'</p><svg width="650" height="70" role="img" aria-label="One snapshot process in this workflow"><rect x="5" y="10" width="600" height="25" fill="#4b7190"/><text x="5" y="58">RUN_SNAPSHOT — '+state+' — '+record.duration+'</text></svg>'
        report.toFile().text='<!doctype html><meta charset="utf-8"><h1>Operational workflow report</h1>'+ 
            '<p>Explicit portable monitoring. CPU, memory and I/O metrics are unavailable. This is not the standard Nextflow resource report.</p><pre>'+ 
            JsonOutput.prettyPrint(JsonOutput.toJson(record)).replace('&','&amp;').replace('<','&lt;')+'</pre>'
    }
}
