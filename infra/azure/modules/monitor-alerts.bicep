param location string
param resourcePrefix string
param apiAppResourceId string
param actionGroupEmail string

resource actionGroup 'Microsoft.Insights/actionGroups@2023-01-01' = {
  name: 'ag-${resourcePrefix}'
  location: 'global'
  properties: {
    groupShortName: take('erpai${uniqueString(resourcePrefix)}', 12)
    enabled: true
    emailReceivers: [
      {
        name: 'ops-email'
        emailAddress: actionGroupEmail
        useCommonAlertSchema: true
      }
    ]
  }
}

resource healthAlert 'Microsoft.Insights/metricAlerts@2018-03-01' = {
  name: 'alert-api-replicas-${resourcePrefix}'
  location: 'global'
  properties: {
    description: 'ERP AI API has zero running replicas'
    severity: 2
    enabled: true
    scopes: [
      apiAppResourceId
    ]
    evaluationFrequency: 'PT5M'
    windowSize: 'PT15M'
    criteria: {
      'odata.type': 'Microsoft.Azure.Monitor.SingleResourceMultipleMetricCriteria'
      allOf: [
        {
          name: 'ReplicaCount'
          metricNamespace: 'Microsoft.App/containerApps'
          metricName: 'Replicas'
          operator: 'LessThan'
          threshold: 1
          timeAggregation: 'Average'
          criterionType: 'StaticThresholdCriterion'
        }
      ]
    }
    actions: [
      {
        actionGroupId: actionGroup.id
      }
    ]
  }
}

output actionGroupId string = actionGroup.id
