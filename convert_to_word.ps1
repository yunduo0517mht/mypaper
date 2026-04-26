# MD to DOCX converter for paper directory
$pandoc = "C:\Users\pytpeng\AppData\Local\Microsoft\WinGet\Packages\JohnMacFarlane.Pandoc_Microsoft.Winget.Source_8wekyb3d8bbwe\pandoc-3.9.0.2\pandoc.exe"
$paperDir = "e:\school\mypaper\paper"

$mdFiles = Get-ChildItem -Path $paperDir -Filter "*.md"

if ($mdFiles.Count -eq 0) {
    Write-Host "No .md files found in $paperDir"
    exit
}

foreach ($file in $mdFiles) {
    $output = [System.IO.Path]::ChangeExtension($file.FullName, ".docx")
    Write-Host "Converting: $($file.Name) -> $([System.IO.Path]::GetFileName($output))"
    & $pandoc $file.FullName -o $output --mathml
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  [OK] Done" -ForegroundColor Green
    } else {
        Write-Host "  [FAIL] Error occurred" -ForegroundColor Red
    }
}

Write-Host "`nAll conversions complete. Files saved to: $paperDir"
