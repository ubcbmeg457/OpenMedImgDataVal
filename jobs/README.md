# Job Scripts

SLURM batch scripts for running pipelines on Alliance Canada / Fir cluster.

## Synapse Authentication

The MRI segmentation pipeline downloads data from Synapse and requires authentication. Set this up before submitting jobs.

1. Copy `.env.example` to `.env` at the project root and fill in your token:

   ```bash
   cp .env.example .env
   ```

2. Generate a personal access token at [synapse.org](https://www.synapse.org/) under **Account Settings > Personal Access Tokens**. Ensure the token has download permissions for the required dataset.

3. Add your token to `.env`:

   ```
   SYNAPSE_AUTH_TOKEN=<your-synapse-personal-access-token>
   ```
## Submitting Jobs

```bash
sbatch jobs/xray-class-shap.sh
sbatch jobs/xray-class-ot.sh
```

## Monitoring

```bash
squeue -u $USER                    # pending/running jobs
scancel <job_id>                   # cancel a job
tail -f xray-class-shap-*.out     # live output
```

## Cluster Resource Commands

```bash
# Account and allocation info
sacctmgr show associations where user=$USER format=account,partition,qos,maxjobs,maxsubmit
sacctmgr show associations where user=$USER format=account%30

# Fairshare and priority
sshare -u $USER

# Partition info (time limits, node counts, GPUs)
scontrol show partition gpubase_bygpu_b2
sinfo -p gpubase_bygpu_b2
sinfo -p gpubase_bygpu_b2 -o "%N %G %m %c"    # nodes, GPUs, memory, CPUs

# Past job history and efficiency
sacct -u $USER --starttime=$(date -d '7 days ago' +%Y-%m-%d) --format=JobID,JobName,Elapsed,MaxRSS,MaxVMSize,State
seff <job_id>                                   # efficiency report for a completed job
```
