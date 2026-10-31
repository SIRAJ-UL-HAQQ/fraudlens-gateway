import js from '@eslint/js';
import globals from 'globals';
import reactHooks from 'eslint-plugin-react-hooks';
import reactRefresh from 'eslint-plugin-react-refresh';

export default [
  { ignores: ['dist'] },
  {
    files: ['**/*.{js,jsx}'],
    languageOptions: {
      ecmaVersion: 2020,
      globals: globals.browser,
      parserOptions: {
        ecmaVersion: 'latest',
        ecmaFeatures: { jsx: true },
        sourceType: 'module',
      },
    },
    plugins: {
      'react-hooks': reactHooks,
      'react-refresh': reactRefresh,
    },
    rules: {
      ...js.configs.recommended.rules,
      ...reactHooks.configs.recommended.rules,
      // Workaround explanation:
      // 1. 'react-hooks' recommended rules includes rules-of-hooks (error) and exhaustive-deps (warn).
      //    Dashboard's useEffect(..., []) does not trigger exhaustive-deps because refreshData() only references
      //    stable state setters (setMetrics, setTransactions), module-level imports, and globals, with no reactive variables.
      // 2. 'no-unused-vars' uses varsIgnorePattern '^[A-Z_]' to ignore React component identifiers in JSX.
      // 3. Warnings do not fail the lint job because npm run lint runs eslint without --max-warnings 0.
      'no-unused-vars': ['warn', { varsIgnorePattern: '^[A-Z_]' }],
      'react-refresh/only-export-components': [
        'warn',
        { allowConstantExport: true },
      ],
    },
  },
];
