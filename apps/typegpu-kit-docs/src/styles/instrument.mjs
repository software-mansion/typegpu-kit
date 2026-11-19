// The site's code colours, for Expressive Code blocks and the examples' source pages: the mark's
// lilac for keywords, its blue for calls and the wordmark's ice for types

// Expressive Code requires a hex background to check contrast
export const instrument = {
  name: 'instrument',
  type: /** @type {const} */ ('dark'),
  colors: {
    'editor.background': '#0a0d12',
    'editor.foreground': '#c9d2dc',
    'editor.selectionBackground': '#4d79ff40',
    'editorLineNumber.foreground': '#56616e',
    'editorLineNumber.activeForeground': '#8d99a6',
  },
  tokenColors: [
    { settings: { foreground: '#c9d2dc', background: '#0a0d12' } },
    {
      scope: ['comment', 'punctuation.definition.comment', 'string.comment'],
      settings: { foreground: '#7c8794', fontStyle: 'italic' },
    },
    {
      scope: ['keyword', 'storage', 'storage.type', 'storage.modifier'],
      settings: { foreground: '#d2a6fb' },
    },
    {
      scope: [
        'keyword.operator',
        'punctuation',
        'meta.brace',
        'punctuation.definition.template-expression',
      ],
      settings: { foreground: '#8592a0' },
    },
    {
      scope: ['string', 'string.template', 'punctuation.definition.string'],
      settings: { foreground: '#6cd0a5' },
    },
    {
      scope: ['constant.numeric', 'constant.language', 'constant.character', 'keyword.other.unit'],
      settings: { foreground: '#6cd0a5' },
    },
    {
      scope: [
        'entity.name.function',
        'support.function',
        'meta.function-call entity.name.function',
      ],
      settings: { foreground: '#8daeff' },
    },
    {
      scope: [
        'entity.name.type',
        'entity.name.class',
        'support.type',
        'support.class',
        'entity.other.inherited-class',
        'meta.type.annotation entity.name.type',
      ],
      settings: { foreground: '#9fd0f5' },
    },
    {
      scope: ['variable.other.object', 'variable.other.constant', 'variable.parameter'],
      settings: { foreground: '#d6dee6' },
    },
    {
      scope: ['variable.other.property', 'meta.object-literal.key', 'support.variable.property'],
      settings: { foreground: '#b4bfcb' },
    },
    { scope: ['markup.inserted'], settings: { foreground: '#7fd1a8' } },
    { scope: ['markup.deleted'], settings: { foreground: '#f08a80' } },
  ],
};
