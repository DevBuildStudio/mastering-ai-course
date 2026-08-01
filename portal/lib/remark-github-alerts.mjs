import { visit } from 'unist-util-visit'
import * as acorn from 'acorn'

const CALLOUT_IMPORT = "import { Callout } from 'nextra/components'\n"

// Maps GitHub-style alert markers (> [!NOTE]) to Nextra's <Callout> props.
const ALERT_TYPES = {
  NOTE: { type: 'info' },
  TIP: { type: 'default', emoji: '💡' },
  IMPORTANT: { type: 'info', emoji: '❗' },
  WARNING: { type: 'warning' },
  CAUTION: { type: 'error', emoji: '⛔' },
}

const MARKER_RE = /^\[!(NOTE|TIP|IMPORTANT|WARNING|CAUTION)\]\s*/

/**
 * Transforms GitHub-flavored alert blockquotes (`> [!NOTE] ...`) into Nextra
 * <Callout> elements so they render with an icon instead of literal text.
 */
export function remarkGithubAlerts() {
  return (tree) => {
    let usedCallout = false

    visit(tree, 'blockquote', (node) => {
      const firstParagraph = node.children?.[0]
      if (!firstParagraph || firstParagraph.type !== 'paragraph') return

      const firstChild = firstParagraph.children?.[0]
      if (!firstChild || firstChild.type !== 'text') return

      const match = MARKER_RE.exec(firstChild.value)
      if (!match) return

      const alert = ALERT_TYPES[match[1]]
      firstChild.value = firstChild.value.slice(match[0].length)
      if (firstChild.value === '') {
        firstParagraph.children.shift()
      }
      if (firstParagraph.children.length === 0) {
        node.children.shift()
      }

      usedCallout = true
      node.type = 'mdxJsxFlowElement'
      node.name = 'Callout'
      node.attributes = [
        { type: 'mdxJsxAttribute', name: 'type', value: alert.type },
        ...(alert.emoji
          ? [{ type: 'mdxJsxAttribute', name: 'emoji', value: alert.emoji }]
          : []),
      ]
    })

    if (usedCallout) {
      tree.children.unshift({
        type: 'mdxjsEsm',
        value: CALLOUT_IMPORT,
        data: {
          estree: acorn.parse(CALLOUT_IMPORT, {
            sourceType: 'module',
            ecmaVersion: 'latest',
          }),
        },
      })
    }
  }
}
