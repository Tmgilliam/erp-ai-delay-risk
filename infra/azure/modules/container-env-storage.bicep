param environmentName string
param storageAccountName string
param fileShareName string
param accessMode string = 'ReadWrite'

resource environment 'Microsoft.App/managedEnvironments@2024-03-01' existing = {
  name: environmentName
}

resource storageAccount 'Microsoft.Storage/storageAccounts@2023-01-01' existing = {
  name: storageAccountName
}

var storageAccountKeys = storageAccount.listKeys()
var storageAccountKey = storageAccountKeys.keys[0].value

resource envStorage 'Microsoft.App/managedEnvironments/storages@2024-03-01' = {
  parent: environment
  name: fileShareName
  properties: {
    azureFile: {
      accountName: storageAccountName
      accountKey: storageAccountKey
      shareName: fileShareName
      accessMode: accessMode
    }
  }
}

output storageName string = envStorage.name
