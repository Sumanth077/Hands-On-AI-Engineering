# Process Resolution Request Workflow

Create this workflow in the same n8n project as the agent and Data Tables. Save it before adding it to the agent as a tool.

## Inputs

Start with **When Executed by Another Workflow** (also called **Execute Sub-workflow Trigger**). Set **Input data mode** to **Define using fields below** and add these string fields:

- `order_id`
- `customer_email`
- `resolution_type`
- `customer_confirmation`
- `agent_note`

## Nodes

### 1. Get the order

Add a **Data Table** node after the trigger.

| Setting | Value |
| --- | --- |
| Resource | Row |
| Operation | Get |
| Data table | `customer_orders` |
| Match | `order_id` equals the trigger's `order_id` |

### 2. Check the order and requested resolution

Add an **If** node. Set its condition to **is true** and use this expression as the value:

```text
{{ $json.customer_email === $('When Executed by Another Workflow').first().json.customer_email && $json.eligible_resolutions.split(';').includes($('When Executed by Another Workflow').first().json.resolution_type) }}
```

### 3. Insert the request

On the **true** branch, add another **Data Table** node.

| Setting | Value |
| --- | --- |
| Resource | Row |
| Operation | Insert |
| Data table | `resolution_requests` |
| Mapping mode | Map each column manually |

Map the table columns from the trigger inputs. Set `status` to the fixed text `pending_review`.

| Column | Value |
| --- | --- |
| `order_id` | Trigger `order_id` |
| `customer_email` | Trigger `customer_email` |
| `resolution_type` | Trigger `resolution_type` |
| `customer_confirmation` | Trigger `customer_confirmation` |
| `status` | `pending_review` |
| `agent_note` | Trigger `agent_note` |

Add an **Edit Fields** node after the insert with:

```text
submitted: true
message: Resolution request recorded for review.
```

### 4. Return an unsupported-resolution result

On the **false** branch of the If node, add an **Edit Fields** node with:

```text
submitted: false
message: The request could not be recorded because the order details or requested resolution did not match.
```

The final node on each branch returns the result to the agent.

## Add It to the Agent

In the Agent Builder, select **Capabilities → Add tools → Workflows**, then choose `Process Resolution Request`. Copy the workflow description from [../agent/tool_contracts.md](../agent/tool_contracts.md).
