param name string
param location string
param environmentId string
param containerImage string
param targetPort int
param cpu string = '0.5'
param memory string = '1Gi'
param minReplicas int = 0
param maxReplicas int = 2

@description('Plain environment variables')
param envVars array = []

@description('Secret-backed environment variables')
param secretEnvVars array = []

@description('Container app secrets')
param secrets array = []

@description('Volume definitions')
param volumes array = []

@description('Volume mounts')
param volumeMounts array = []

var containerEnvEntries = [for item in envVars: {
  name: item.name
  value: item.value
}]

var containerSecretEntries = [for item in secretEnvVars: {
  name: item.name
  secretRef: item.secretRef
}]

var allEnv = concat(containerEnvEntries, containerSecretEntries)

resource containerApp 'Microsoft.App/containerApps@2024-03-01' = {
  name: name
  location: location
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    managedEnvironmentId: environmentId
    configuration: {
      ingress: {
        external: true
        targetPort: targetPort
        transport: 'auto'
        allowInsecure: false
      }
      secrets: secrets
    }
    template: {
      containers: [
        {
          name: name
          image: containerImage
          env: allEnv
          volumeMounts: volumeMounts
          resources: {
            cpu: json(cpu)
            memory: memory
          }
        }
      ]
      scale: {
        minReplicas: minReplicas
        maxReplicas: maxReplicas
      }
      volumes: volumes
    }
  }
}

output id string = containerApp.id
output name string = containerApp.name
output fqdn string = containerApp.properties.configuration.ingress.fqdn
output principalId string = containerApp.identity.principalId
