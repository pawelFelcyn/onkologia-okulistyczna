param(
    [string]$Email
)

if (-not $Email) {
    $envFile = Join-Path (Split-Path $MyInvocation.MyCommand.Path) ".env"

    if (-not (Test-Path $envFile)) {
        Write-Error ".env file not found and no email parameter provided."
        exit 1
    }

    $envContent = Get-Content $envFile | Where-Object { $_ -match '^NOTIFY_EMAIL=' }

    if (-not $envContent) {
        Write-Error "NOTIFY_EMAIL not found in .env and no email parameter provided."
        exit 1
    }

    $Email = $envContent -replace '^NOTIFY_EMAIL=', ''
}

Write-Host "Using email: $Email"

$remoteHost = "access.cluster.wmi.amu.edu.pl"

$remoteDir = "/projects/onkokul/onkologia-okulistyczna/cluster_scripts"
$remoteScript = "train_unet.sh"

ssh $remoteHost "cd $remoteDir && sbatch $remoteScript --mail-user=""$Email"""