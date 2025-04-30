# fetch_corpus.ps1  (PowerShell version)
$Target = "data/raw"
New-Item -ItemType Directory -Path $Target -Force | Out-Null

$links = @{
  "https://www.ibm.com/investor/att/pdf/IBM_Business_Conduct_Guidelines.pdf" = "ibm_business_conduct_guidelines.pdf"
  "https://cdn.pfizer.com/pfizercom/investors/corporate/Pfizer_2023BlueBook_English_v5_02272024.pdf" = "pfizer_code_of_conduct_2023.pdf"
  "https://www.walmartethics.com/content/dam/walmartethics/documents/code_of_conduct/Code_of_Conduct_English_US.pdf" = "walmart_code_of_conduct.pdf"
  "https://www.goldmansachs.com/investor-relations/corporate-governance/corporate-governance-documents/code-of-business-conduct-and-ethics.pdf" = "goldman_sachs_code_of_conduct.pdf"
  "https://www.chevron.com/-/media/shared-media/documents/chevronbusinessconductethicscode.pdf" = "chevron_ethics_code.pdf"
  "https://www.nestle.com/sites/default/files/asset-library/documents/library/documents/corporate_governance/corporate-business-principles-en.pdf" = "nestle_corporate_business_principles_2025.pdf"
  "https://www.accenture.com/content/dam/accenture/final/a-com-migration/pdf/pdf-63/accenture-cobe-brochure-english.pdf" = "accenture_code_of_business_ethics.pdf"
  "https://www.ethics.org/wp-content/uploads/resources/Boeing-Ethics1.pdf" = "boeing_conduct_guidelines.pdf"
  "https://www.unitedhealthgroup.com/content/dam/UHG/PDF/About/UNH-Code-of-Conduct.pdf" = "unitedhealth_group_code_of_conduct.pdf"
  "https://cdn.fastly.steamstatic.com/apps/valve/Valve_NewEmployeeHandbook.pdf" = "valve_employee_handbook.pdf"
  "https://cdn2.hubspot.net/hubfs/191357/Webinar%20Copy%20-%20%20Remote%20Work%20Policy%20(1).pdf" = "vanderbloemen_remote_work_policy.pdf"
  "https://www.apple.com/hk/supplier-responsibility/pdf/Apple-Supplier-Code-of-Conduct-and-Supplier-Responsibility-Standards.pdf" = "apple_supplier_code_of_conduct.pdf"
  "https://www.lobbyregister.bundestag.de/media/fd/49/396292/Lobbyregister-Google-Code-of-Conduct.pdf" = "google_code_of_conduct.pdf"
  "https://content-prod-live.cert.starbucks.com/binary/v2/asset/137-95141.pdf" = "starbucks_standards_of_conduct.pdf"
  "https://www.morganstanley.com/content/dam/msdotcom/en/assets/pdfs/Code_of_Conduct_Morgan_Stanley_2023.pdf" = "morgan_stanley_code_of_conduct_2023.pdf"
}

foreach ($url in $links.Keys) {
  $file = Join-Path $Target $links[$url]
  Write-Host "→ $file"
  Invoke-WebRequest -Uri $url -OutFile $file -UseBasicParsing -Headers @{ "User-Agent" = "Mozilla/5.0" }
}

Write-Host "Done. $(Get-ChildItem $Target).Count files downloaded."

