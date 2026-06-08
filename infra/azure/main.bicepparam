using 'main.bicep'

param environmentName = 'erpai'
param location = 'eastus2'
param deployApim = false
param dashboardUser = 'demo@erp-ai.local'
// Override at deploy time: param dashboardPassword = readEnvironmentVariable('DASHBOARD_PASSWORD', 'ChangeMe-Demo1234!')
param dashboardPassword = 'ChangeMe-Demo1234!'
