# Tool Contracts

Configure these as separate tools in the n8n Agent Builder. They are intentionally narrow: one reads demo order data and one creates a request after approval.

## `lookup_customer_order`

| Setting | Value |
| --- | --- |
| Integration | n8n Data Table |
| Resource | Data table |
| Operation | Must Match |
| Table | `customer_orders` |
| Sensitive approval | Off |
| Input expected from agent | `order_id` |
| Output the agent needs | Order status, items, delivery status, eligible resolutions, and notes |

Use this description in n8n:

> Look up one customer order by its exact order ID. Call this only after the customer gives an order ID or asks about a specific order. This tool is read-only. Use its result to confirm facts before explaining a policy or proposing a resolution.

Configure the filter so `order_id` matches the order ID supplied by the agent. Do not expose a tool that retrieves every order row.

## `create_resolution_request`

| Setting | Value |
| --- | --- |
| Integration | n8n Data Table |
| Resource | Data table |
| Operation | Insert |
| Table | `resolution_requests` |
| Sensitive approval | On |
| Inputs expected from agent | `order_id`, `customer_email`, `resolution_type`, `customer_confirmation`, `status`, and `agent_note` |

Use this description in n8n:

> Create one internal resolution request after the order has been found, the policy supports the resolution, and the customer has explicitly confirmed what they want. This tool must require approval. Set `status` to `pending_review`. Do not use this tool for general questions, policy explanations, or uncertain cases.

Set `resolution_type` to one of: `replacement`, `return_review`, `delivery_investigation`, or `human_escalation`.

The tool records a request. It does not issue a refund, change an order, or contact a courier.
