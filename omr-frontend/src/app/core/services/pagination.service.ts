import { Injectable } from '@angular/core';

@Injectable({
  providedIn: 'root',
})
export class PaginationService {
  paginate<T>(data: T[], pageIndex: number, pageSize: number): T[] {
    const start = pageIndex * pageSize;
    return data.slice(start, start + pageSize);
  }

  totalPages(totalItems: number, pageSize: number): number {
    return Math.max(1, Math.ceil(totalItems / pageSize));
  }

  rangeLabel(totalItems: number, pageIndex: number, pageSize: number, label: string = 'items'): string {
    if (totalItems === 0) {
      return `0 ${label}`;
    }

    const start = pageIndex * pageSize + 1;
    const end = Math.min(start + pageSize - 1, totalItems);

    return `${start}–${end} of ${totalItems} ${label}`;
  }
}
