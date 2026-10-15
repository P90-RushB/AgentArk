# SimpleAgent record and note tools

When SimpleAgent filesystem tools are enabled, their native function declarations
explain the workspace and include example calls. The guidance belongs to those
tools; it is not added when filesystem tools are disabled or a registry contains
only custom tools. Host tools inspect visible history and model notes. They do not
execute environment actions; finish a native-tool decision with `submit_action`
according to its own declaration.

The workspace exposes logical paths:

| Path | Contents | Access |
| --- | --- | --- |
| `records/<record_id>/messages.json` | Original visible input messages and image references | Read-only |
| `records/<record_id>/assistant.txt` | Provider final response text, which can be empty for native submissions | Read-only |
| `records/<record_id>/action.txt` | Extracted environment action | Read-only |
| `records/<record_id>/images/<number>.<extension>` | Original observation image bytes, when present | Read-only |
| `notes/<name>` | Model-authored UTF-8 memory, with one filename and no nested directories | Read/write |

`records/` is a virtual archive backed by immutable in-memory snapshots, not an
OS directory. Notes use a restricted temporary workspace. Arbitrary OS paths are
not exposed. A completed decision publishes its input and selected response;
the current decision's input is not available as a record until it completes.

Tool declarations count toward the existing context budgets. Very small custom
budgets need room for this help as well as the required input; the runtime limits
and shared defaults are unchanged.

Compaction replaces older working-context messages and images with summaries.
It does not delete their published snapshots. An old image can therefore be
reopened as actual vision content with `view_image`, provided it was published,
is still in the current tool scope, and fits the call/image/byte limits. Listing
is the authority for available paths. Environment attempt transitions can carry
history according to the runtime policy; `SimpleAgent.reset()` starts a new case
and `SimpleAgent.close()` ends the workspace. This archive is not durable storage.

Start by discovering paths with a native function call:

```text
list_files({"path":"","offset":0,"limit":20})
```

The result contains flat file rows, not directory entries. To filter, pass
`"records"`, `"notes"`, or a discovered record root without a trailing slash.
Continue with the returned `next_offset` and the same filter until it is `null`.

For example, after listing confirms the record paths, read/view them and create
a note. Replace these record paths with actual returned paths rather than
guessing record IDs or image numbers. Choose your own single-level note filename:

```text
read_file({"path":"records/turn-000001/messages.json","start_line":1,"max_lines":100})
view_image({"path":"records/turn-000001/images/0000.png"})
write_note({"path":"notes/plan.md","content":"Recheck the earlier reference image before acting."})
```

`read_file` reads UTF-8 text by 1-based lines. Follow `next_line` until `null` to
read more. In `messages.json`, image paths are relative to that record's root;
concatenate any `text_chunks` exactly without a separator. Use `view_image` for
image bytes. `write_note` creates or fully replaces a note; notes and summaries
are model interpretations, not new environment facts.
