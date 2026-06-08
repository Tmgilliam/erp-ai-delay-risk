param name string
param location string

resource workspace 'Microsoft.OperationalInsights/workspaces@2022-10-01' = {
  name: name
  location: location
  properties: {
    sku: {
      name: 'PerGB2018'
    }
    retentionInDays: 30
  }
}

output customerId string = workspace.properties.customerId
output primarySharedKey string = workspace.listKeys().primarySharedKey
output id string = workspace.id
