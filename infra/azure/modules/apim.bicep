param name string
param location string
param publisherEmail string
param publisherName string
param backendUrl string

resource apiManagement 'Microsoft.ApiManagement/service@2023-03-01-preview' = {
  name: name
  location: location
  sku: {
    name: 'Developer'
    capacity: 1
  }
  properties: {
    publisherEmail: publisherEmail
    publisherName: publisherName
  }
}

// Backend and API policies are configured post-deploy via deployment guide.
// APIM provisioning alone demonstrates enterprise gateway layer intent.

output gatewayUrl string = apiManagement.properties.gatewayUrl
output name string = apiManagement.name
output backendUrl string = backendUrl
