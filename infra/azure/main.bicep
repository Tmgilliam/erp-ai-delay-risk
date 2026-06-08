targetScope = 'resourceGroup'

@description('Environment name prefix for resources')
param environmentName string = 'erpai'

@description('Azure region for all resources')
param location string = resourceGroup().location

@description('Deploy Azure API Management (Developer tier — additional cost)')
param deployApim bool = false

@description('Dashboard login email stored in Key Vault')
param dashboardUser string = 'demo@erp-ai.local'

@secure()
@description('Dashboard login password stored in Key Vault')
param dashboardPassword string

@description('Placeholder image until ACR images are pushed post-deploy')
param placeholderImage string = 'mcr.microsoft.com/azuredocs/containerapps-helloworld:latest'

var resourcePrefix = '${environmentName}${uniqueString(resourceGroup().id)}'
var acrName = replace('cr${resourcePrefix}', '-', '')
var storageAccountName = replace('st${resourcePrefix}', '-', '')
var keyVaultName = 'kv-${take(resourcePrefix, 12)}'
var logAnalyticsName = 'log-${resourcePrefix}'
var containerEnvName = 'cae-${resourcePrefix}'
var apiAppName = 'ca-api-${resourcePrefix}'
var dashboardAppName = 'ca-dash-${resourcePrefix}'
var apimName = 'apim-${resourcePrefix}'

module logAnalytics 'modules/log-analytics.bicep' = {
  name: 'logAnalytics'
  params: {
    name: logAnalyticsName
    location: location
  }
}

module acr 'modules/acr.bicep' = {
  name: 'acr'
  params: {
    name: acrName
    location: location
  }
}

module storage 'modules/storage.bicep' = {
  name: 'storage'
  params: {
    name: storageAccountName
    location: location
  }
}

module keyVault 'modules/keyvault.bicep' = {
  name: 'keyVault'
  params: {
    name: keyVaultName
    location: location
    dashboardUser: dashboardUser
    dashboardPassword: dashboardPassword
  }
}

module containerEnv 'modules/container-env.bicep' = {
  name: 'containerEnv'
  params: {
    name: containerEnvName
    location: location
    logAnalyticsCustomerId: logAnalytics.outputs.customerId
    logAnalyticsSharedKey: logAnalytics.outputs.primarySharedKey
  }
}

module storageEnv 'modules/container-env-storage.bicep' = {
  name: 'storageEnv'
  params: {
    environmentName: containerEnvName
    storageAccountName: storage.outputs.name
    fileShareName: 'scoring-history'
    accessMode: 'ReadWrite'
  }
  dependsOn: [
    containerEnv
    storage
  ]
}

module apiApp 'modules/container-app.bicep' = {
  name: 'apiApp'
  dependsOn: [
    storageEnv
  ]
  params: {
    name: apiAppName
    location: location
    environmentId: containerEnv.outputs.id
    containerImage: placeholderImage
    targetPort: 8000
    cpu: '1.0'
    memory: '2Gi'
    minReplicas: 0
    maxReplicas: 3
    envVars: [
      {
        name: 'AZURE_STORAGE_ACCOUNT_NAME'
        value: storage.outputs.name
      }
      {
        name: 'MODEL_BLOB_CONTAINER'
        value: 'models'
      }
      {
        name: 'MODEL_BLOB_NAME'
        value: 'delay_model.pkl'
      }
      {
        name: 'MODEL_PATH'
        value: '/tmp/delay_model.pkl'
      }
    ]
    volumeMounts: [
      {
        volumeName: 'scoring-history'
        mountPath: '/app/monitoring'
      }
    ]
    volumes: [
      {
        name: 'scoring-history'
        storageType: 'AzureFile'
        storageName: 'scoring-history'
      }
    ]
  }
}

module dashboardApp 'modules/container-app.bicep' = {
  name: 'dashboardApp'
  params: {
    name: dashboardAppName
    location: location
    environmentId: containerEnv.outputs.id
    containerImage: placeholderImage
    targetPort: 8501
    cpu: '0.5'
    memory: '1Gi'
    minReplicas: 0
    maxReplicas: 2
    envVars: [
      {
        name: 'API_URL'
        value: 'https://${apiApp.outputs.fqdn}'
      }
      {
        name: 'DASH_ROLE'
        value: 'EXEC'
      }
    ]
    secretEnvVars: [
      {
        name: 'DASH_USER'
        secretRef: 'dash-user'
      }
      {
        name: 'DASH_PASS'
        secretRef: 'dash-pass'
      }
    ]
    secrets: [
      {
        name: 'dash-user'
        value: dashboardUser
      }
      {
        name: 'dash-pass'
        value: dashboardPassword
      }
    ]
  }
}

module apiAcrPull 'modules/acr-pull-role.bicep' = {
  name: 'apiAcrPull'
  params: {
    acrName: acr.outputs.name
    principalId: apiApp.outputs.principalId
  }
}

module dashboardAcrPull 'modules/acr-pull-role.bicep' = {
  name: 'dashboardAcrPull'
  params: {
    acrName: acr.outputs.name
    principalId: dashboardApp.outputs.principalId
  }
}

module apiStorageRole 'modules/storage-blob-role.bicep' = {
  name: 'apiStorageRole'
  params: {
    storageAccountName: storage.outputs.name
    principalId: apiApp.outputs.principalId
  }
}

module alerts 'modules/monitor-alerts.bicep' = {
  name: 'alerts'
  params: {
    location: location
    resourcePrefix: resourcePrefix
    apiAppResourceId: apiApp.outputs.id
    actionGroupEmail: dashboardUser
  }
}

module apim 'modules/apim.bicep' = if (deployApim) {
  name: 'apim'
  params: {
    name: apimName
    location: location
    publisherEmail: dashboardUser
    publisherName: 'ERP AI Delay Risk'
    backendUrl: 'https://${apiApp.outputs.fqdn}'
  }
}

output acrLoginServer string = acr.outputs.loginServer
output acrName string = acr.outputs.name
output storageAccountName string = storage.outputs.name
output keyVaultName string = keyVault.outputs.name
output apiUrl string = 'https://${apiApp.outputs.fqdn}'
output dashboardUrl string = 'https://${dashboardApp.outputs.fqdn}'
output apimGatewayUrl string = deployApim ? apim.outputs.gatewayUrl : ''
output apiAppName string = apiApp.outputs.name
output dashboardAppName string = dashboardApp.outputs.name
output apiPrincipalId string = apiApp.outputs.principalId
output dashboardPrincipalId string = dashboardApp.outputs.principalId
