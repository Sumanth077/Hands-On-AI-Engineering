# Tool Contracts

The agent has one narrow read tool and one workflow tool. It does not receive direct write access to `resolution_requests`.

## `lookup_customer_order`

| Setting | Value |
| --- | --- |
| Integration | n8n Data Table |
| Resource | Data table |
| Operation | Must Match |
| Table | `customer_orders` |
| Sensitive approval | Off |
| Input | `order_id` |

Use this description in n8n:

> Look up one customer order by its exact order ID. Call this only after the customer gives an order ID or asks about a specific order. This tool is read-only. Use its result to confirm facts before explaining a policy or proposing a resolution.

Configure the filter so `order_id` matches the value supplied by the agent. Do not expose a tool that retrieves every order row.

## `Process Resolution Request`

| Setting | Value |
| --- | --- |
| Tool type | Workflow |
| Workflow | `Process Resolution Request` |
| Sensitive approval | Off for this demo |
| Inputs | `order_id`, `customer_email`, `resolution_type`, `customer_confirmation`, `agent_note` |

Use this description in n8n:

> Record an internal resolution request after the exact order has been found, the policy supports the resolution, and the customer has explicitly confirmed what they want. This workflow validates the requested resolution against the order and creates a pending_review record. It does not issue refunds, ship replacements, change orders, or contact the customer.

The workflow—not the agent—sets `status` to `pending_review`. The supported resolution types are `replacement`, `return_review`, `delivery_investigation`, and `human_escalation`.
