import React from 'react';
import Skeleton from './Skeleton';

export function DataTable({
  columns = [],
  data = [],
  keyField = 'id',
  isLoading = false,
  emptyMessage = 'No data available',
  onRowClick = null,
  className = '',
}) {
  return (
    <div className={`overflow-x-auto rounded-xl border border-slate-800 bg-slate-900 ${className}`}>
      <table className="w-full text-left text-xs border-collapse">
        <thead className="bg-slate-950/70 border-b border-slate-800 text-slate-400 uppercase tracking-wider font-semibold">
          <tr>
            {columns.map((col) => (
              <th
                key={col.key}
                scope="col"
                className={`py-3.5 px-4 ${col.className || ''}`}
              >
                {col.header}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="divide-y divide-slate-800/60 text-slate-200">
          {isLoading ? (
            Array.from({ length: 5 }).map((_, rIdx) => (
              <tr key={`skeleton-row-${rIdx}`}>
                {columns.map((col) => (
                  <td key={`skeleton-col-${col.key}`} className="py-3.5 px-4">
                    <Skeleton className="h-4 w-full max-w-[120px]" />
                  </td>
                ))}
              </tr>
            ))
          ) : data.length === 0 ? (
            <tr>
              <td
                colSpan={columns.length}
                className="py-12 px-4 text-center text-slate-500 italic"
              >
                {emptyMessage}
              </td>
            </tr>
          ) : (
            data.map((row) => {
              const rowKey = row[keyField] ?? JSON.stringify(row);
              return (
                <tr
                  key={rowKey}
                  onClick={() => onRowClick && onRowClick(row)}
                  className={`transition-colors ${
                    onRowClick ? 'hover:bg-slate-800/50 cursor-pointer' : 'hover:bg-slate-800/30'
                  }`}
                >
                  {columns.map((col) => (
                    <td
                      key={`${rowKey}-${col.key}`}
                      className={`py-3.5 px-4 ${col.className || ''}`}
                    >
                      {col.render ? col.render(row) : String(row[col.key] ?? '')}
                    </td>
                  ))}
                </tr>
              );
            })
          )}
        </tbody>
      </table>
    </div>
  );
}

export default DataTable;
