import { defineComponent, h } from 'vue'

function icon(paths: ReturnType<typeof h>[], size = 15) {
  return defineComponent({
    render: () => h('svg', {
      width: size, height: size,
      viewBox: '0 0 24 24',
      fill: 'none',
      stroke: 'currentColor',
      'stroke-width': '2',
    }, paths),
  })
}

export const SelectIcon = icon([
  h('path', { d: 'M5 3l14 9-7 1-3 7-4-17z' }),
])

export const DetectIcon = icon([
  h('circle', { cx: '11', cy: '11', r: '8' }),
  h('line', { x1: '21', y1: '21', x2: '16.65', y2: '16.65' }),
])

export const InpaintIcon = icon([
  h('path', { d: 'M12 19l7-7 3 3-7 7-3-3z' }),
  h('path', { d: 'M18 13l-1.5-7.5L2 2l3.5 14.5L13 18l5-5z' }),
  h('circle', { cx: '11', cy: '11', r: '2' }),
])

export const SwapIcon = icon([
  h('polyline', { points: '17 1 21 5 17 9' }),
  h('path', { d: 'M3 11V9a4 4 0 0 1 4-4h14' }),
  h('polyline', { points: '7 23 3 19 7 15' }),
  h('path', { d: 'M21 13v2a4 4 0 0 1-4 4H3' }),
])
export const SettingsIcon = defineComponent({
  render: () =>
    h('svg', { width: 14, height: 14, viewBox: '0 0 24 24', fill: 'none', stroke: 'currentColor', 'stroke-width': '2.5' }, [
      h('circle', { cx: 12, cy: 12, r: 3 }),
      h('path', { d: 'M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z' }),
    ]),
})
export const ZoomInIcon = icon([
  h('circle', { cx: '11', cy: '11', r: '8' }),
  h('line', { x1: '21', y1: '21', x2: '16.65', y2: '16.65' }),
  h('line', { x1: '11', y1: '8', x2: '11', y2: '14' }),
  h('line', { x1: '8', y1: '11', x2: '14', y2: '11' }),
], 14)

export const ZoomOutIcon = icon([
  h('circle', { cx: '11', cy: '11', r: '8' }),
  h('line', { x1: '21', y1: '21', x2: '16.65', y2: '16.65' }),
  h('line', { x1: '8', y1: '11', x2: '14', y2: '11' }),
], 14)

export const UndoIcon = icon([
  h('polyline', { points: '9 14 4 9 9 4' }),
  h('path', { d: 'M20 20v-7a4 4 0 0 0-4-4H4' }),
], 14)

export const RedoIcon = icon([
  h('polyline', { points: '15 14 20 9 15 4' }),
  h('path', { d: 'M4 20v-7a4 4 0 0 1 4-4h12' }),
], 14)

export const ResetIcon = icon([
  h('polyline', { points: '1 4 1 10 7 10' }),
  h('path', { d: 'M3.51 15a9 9 0 1 0 .49-4.95' }),
], 14)

export const tools = [
  { id: 'select',  label: 'Select',      shortcut: 'V', icon: SelectIcon  },
  { id: 'detect',  label: 'Detect',      shortcut: 'D', icon: DetectIcon  },
  { id: 'inpaint', label: 'Inpaint',     shortcut: 'E', icon: InpaintIcon },
  { id: 'swap',    label: 'Swap object', shortcut: 'R', icon: SwapIcon    },
]