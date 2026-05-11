import { Component, Input, Output, EventEmitter } from '@angular/core';

import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-pagination',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './pagination.html',
  styleUrls: ['./pagination.scss'],
})
export class PaginationComponent {
  @Input() pageIndex = 0;

  @Input() totalPages = 1;

  @Input() pageSize = 5;

  @Input() rangeLabel = '';

  @Output() pageIndexChange = new EventEmitter<number>();

  @Output() pageSizeChange = new EventEmitter<number>();

  prevPage() {
    if (this.pageIndex > 0) {
      this.pageIndexChange.emit(this.pageIndex - 1);
    }
  }

  nextPage() {
    if (this.pageIndex < this.totalPages - 1) {
      this.pageIndexChange.emit(this.pageIndex + 1);
    }
  }

  changePageSize(event: Event) {
    const size = Number((event.target as HTMLSelectElement).value);

    this.pageSizeChange.emit(size);
  }
}
