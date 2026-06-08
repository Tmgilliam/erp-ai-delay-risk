param name string
param location string
param dashboardUser string
@secure()
param dashboardPassword string

resource keyVault 'Microsoft.KeyVault/vaults@2023-07-01' = {
  name: name
  location: location
  properties: {
    sku: {
      family: 'A'
      name: 'standard'
    }
    tenantId: subscription().tenantId
    enableRbacAuthorization: true
    enabledForTemplateDeployment: true
    publicNetworkAccess: 'Enabled'
  }
}

resource dashUserSecret 'Microsoft.KeyVault/vaults/secrets@2023-07-01' = {
  parent: keyVault
  name: 'dashboard-user'
  properties: {
    value: dashboardUser
  }
}

resource dashPassSecret 'Microsoft.KeyVault/vaults/secrets@2023-07-01' = {
  parent: keyVault
  name: 'dashboard-password'
  properties: {
    value: dashboardPassword
  }
}

output name string = keyVault.name
output id string = keyVault.id
