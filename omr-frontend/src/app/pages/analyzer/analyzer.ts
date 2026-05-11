import { Component, OnDestroy, ChangeDetectorRef } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { HttpClient } from '@angular/common/http';
import { Subscription } from 'rxjs';
import { Sidebar } from '../../layout/sidebar/sidebar';
import { Header } from '../../layout/header/header';
import { SidebarStateService } from '../../core/services/sidebar-state.service';
import { PaginationService } from '../../core/services/pagination.service';
import { PaginationComponent } from '../../shared/components/pagination/pagination';

const BASE = 'http://localhost:5000';
const WC = { withCredentials: true };

const LEVEL_LABELS: Record<string, string> = {
  lower_primary: 'Lower Primary',
  higher_primary: 'Higher Primary',
  junior: 'Junior',
  intermediate: 'Intermediate',
  senior: 'Senior',
  open: 'Open',
};

@Component({
  selector: 'app-dashboard',
  standalone: true,
  imports: [CommonModule, FormsModule, Sidebar, Header, PaginationComponent],
  templateUrl: './analyzer.html',
  styleUrls: ['./analyzer.scss'],
})
export class Analyzer implements OnDestroy {
  isSidebarOpen = true;
  private sidebarSub!: Subscription;

  selectedFiles: File[] = [];
  isDragging = false;
  isProcessing = false;
  successMessage = '';
  errorMessage = '';
  progressPct = 0;
  progressText = '';
  liveCount = 0;

  private batchId = '';
  private pollTimer: any = null;
  private offset = 0;

  allResults: Record<string, any> = {};
  resultRows: any[] = [];
  filteredRows: any[] = [];
  searchQuery = '';

  pageIndex = 0;
  pageSize = 5;


  get pagedRows(): any[] {
    return this.paginationService.paginate(this.filteredRows, this.pageIndex, this.pageSize);
  }
  get totalPages(): number {
    return this.paginationService.totalPages(this.filteredRows.length, this.pageSize);
  }
  get rangeLabel(): string {
    return this.paginationService.rangeLabel(
      this.filteredRows.length,
      this.pageIndex,
      this.pageSize,
      'results',
    );
  }
  onPageSizeChange(size: number) {
    this.pageSize = Number(size);
    this.pageIndex = 0;
    this.cdr.detectChanges();
  }

  modalOpen = false;
  modalStudent: any = null;

  constructor(
    private http: HttpClient,
    private cdr: ChangeDetectorRef,
    private sidebarState: SidebarStateService,
    private paginationService: PaginationService,
  ) {
    this.isSidebarOpen = this.sidebarState.isOpen;
    this.sidebarSub = this.sidebarState.isOpen$.subscribe((open) => {
      this.isSidebarOpen = open;
      this.cdr.detectChanges();
    });
  }

  ngOnDestroy(): void {
    this.sidebarSub?.unsubscribe();
    this.stopPolling();
  }

  onFileSelected(event: any) {
    const files: FileList = event.target.files;
    if (!files) return;
    Array.from(files).forEach((file) => {
      if (!this.selectedFiles.some((f) => f.name === file.name)) this.selectedFiles.push(file);
    });
  }

  removeFile(index: number) {
    this.selectedFiles.splice(index, 1);
  }

  onDragOver(event: DragEvent) {
    event.preventDefault();
    this.isDragging = true;
  }
  onDragLeave(event: DragEvent) {
    this.isDragging = false;
  }

  onDrop(event: DragEvent) {
    event.preventDefault();
    this.isDragging = false;
    const files = event.dataTransfer?.files;
    if (!files) return;
    Array.from(files).forEach((file) => {
      if (!this.selectedFiles.some((f) => f.name === file.name)) this.selectedFiles.push(file);
    });
  }

  processFiles() {
    if (!this.selectedFiles.length || this.isProcessing) return;

    this.isProcessing = true;
    this.successMessage = '';
    this.errorMessage = '';
    this.progressPct = 0;
    this.progressText = 'Starting…';
    this.liveCount = 0;
    this.offset = 0;

    this.allResults = {};
    this.resultRows = [];
    this.filteredRows = [];
    this.searchQuery = '';
    this.pageIndex = 0;

    this.http.post<{ batch_id: string }>(`${BASE}/api/start`, {}, WC).subscribe({
      next: (res) => {
        this.batchId = res.batch_id;

        const formData = new FormData();
        this.selectedFiles.forEach((f) => formData.append('files', f));
        formData.append('batch_id', this.batchId);

        this.http.post(`${BASE}/api/upload`, formData, WC).subscribe({
          next: () => console.log('Upload accepted, polling started…'),
          error: (err) => {
            console.error('Upload error:', err);
            this.errorMessage = 'Upload failed. Please try again.';
            this.reset();
          },
        });

        this.startPolling();
      },
      error: (err) => {
        console.error('Start error:', err);
        this.errorMessage = 'Could not start processing. Please log in again.';
        this.isProcessing = false;
        this.cdr.detectChanges();
      },
    });
  }

  private startPolling() {
    this.pollTimer = setInterval(() => this.poll(), 1200);
  }

  private stopPolling() {
    if (this.pollTimer) {
      clearInterval(this.pollTimer);
      this.pollTimer = null;
    }
  }

  private poll() {
    this.http.get<any>(`${BASE}/api/results/${this.batchId}?offset=${this.offset}`, WC).subscribe({
      next: (data) => {
        this.progressPct = data.percent ?? 0;
        this.progressText = data.status ?? 'Processing…';
        this.liveCount += (data.results ?? []).length;
        this.offset = data.offset ?? this.offset;

        this.appendRows(data.results ?? []);

        const status = (data.status ?? '').toLowerCase();

        if (status === 'completed') {
          this.stopPolling();
          this.isProcessing = false;
          this.progressPct = 100;
          this.progressText = 'Completed';
          this.successMessage = `Done — ${this.liveCount} sheet(s) processed.`;
          this.selectedFiles = [];
          this.cdr.detectChanges();
          setTimeout(() => {
            this.successMessage = `Done — ${this.liveCount} sheet(s) processed.`;
            this.cdr.detectChanges();
            setTimeout(() => {
              this.successMessage = '';
              this.cdr.detectChanges();
            }, 3000);
          }, 200);
        }

        if (status === 'failed') {
          this.stopPolling();
          this.errorMessage = 'Processing failed on the server.';
          this.reset();
          this.cdr.detectChanges();
        }
      },
      error: (err) => {
        console.error('Poll error:', err);
        this.stopPolling();
        this.errorMessage = 'Lost connection during processing.';
        this.reset();
        this.cdr.detectChanges();
      },
    });
  }

  private reset() {
    this.isProcessing = false;
    this.progressPct = 0;
    this.progressText = '';
    this.liveCount = 0;
    this.stopPolling();
    this.cdr.detectChanges();
  }

  private appendRows(results: any[]) {
    if (!results.length) return;
    results.forEach((r) => {
      if (this.allResults[r.key]) return;
      this.allResults[r.key] = r;
      this.resultRows.push(r);
    });
    this.applyFilter();
    this.cdr.detectChanges();
  }

  onSearch(query: string) {
    this.searchQuery = query;
    this.pageIndex = 0;
    this.applyFilter();
  }

  private applyFilter() {
    const q = this.searchQuery.toLowerCase();
    this.filteredRows = q
      ? this.resultRows.filter(
          (r) =>
            (r.name || '').toLowerCase().includes(q) ||
            (r.centre_number || '').toLowerCase().includes(q),
        )
      : [...this.resultRows];
  }

  get statTotal(): number {
    return this.resultRows.length;
  }

  get statAvg(): string {
    if (!this.resultRows.length) return '—';
    const avg =
      this.resultRows.map((r) => parseFloat(r.percentage) || 0).reduce((s, v) => s + v, 0) /
      this.resultRows.length;
    return avg.toFixed(1) + '%';
  }

  get statHigh(): string {
    if (!this.resultRows.length) return '—';
    return Math.max(...this.resultRows.map((r) => parseFloat(r.percentage) || 0)).toFixed(1) + '%';
  }

  levelLabel(lv: string): string {
    return LEVEL_LABELS[lv] || lv || '—';
  }
  levelClass(lv: string): string {
    return 'lv-badge lv-' + lv;
  }

  scoreClass(pct: number): string {
    if (pct >= 75) return 'score-g';
    if (pct >= 50) return 'score-y';
    return 'score-r';
  }

  emptyCount(r: any): number {
    if (r.empty !== undefined) return r.empty;
    return (r.answers || []).filter(
      (a: any) => !a.value || (a.value || '').toLowerCase() === 'empty',
    ).length;
  }

  rowIndex(r: any): string {
    return String(this.resultRows.indexOf(r) + 1).padStart(2, '0');
  }
  normalizedLevel(r: any): string {
    return (r.level || '').toLowerCase().replace(/\s+/g, '_');
  }

  viewResult(key: string) {
    const r = this.allResults[key];
    if (!r) return;
    this.modalStudent = r;
    this.modalOpen = true;
  }

  closeModal() {
    this.modalOpen = false;
    this.modalStudent = null;
  }
  nameInitial(name: string): string {
    return name ? name.trim()[0].toUpperCase() : '?';
  }

  modalMetaLine(): string {
    if (!this.modalStudent) return '';
    const lv = this.normalizedLevel(this.modalStudent);
    return [this.modalStudent.centre_number, this.levelLabel(lv), this.modalStudent.dob]
      .filter(Boolean)
      .join('  ·  ');
  }

  modalPctColor(): string {
    const pct = parseFloat(this.modalStudent?.percentage) || 0;
    if (pct >= 75) return 'var(--green)';
    if (pct >= 50) return 'var(--amber)';
    return 'var(--red)';
  }

  modalAnswers(): any[] {
    return this.modalStudent?.answers || [];
  }

  pillClass(a: any): string {
    const val = (a.value || '').trim();
    if (val.toLowerCase().includes('multiple')) return 'pill multi';
    if (!val || val.toLowerCase() === 'empty') return 'pill empty';
    if (a.is_correct) return 'pill correct';
    return 'pill wrong';
  }

  pillDisplay(a: any): string {
    const val = (a.value || '').trim();
    if (!val || val.toLowerCase() === 'empty') return '—';
    return val;
  }

  modalCorrect(): number {
    return this.modalStudent?.correct || 0;
  }
  modalWrong(): number {
    return this.modalStudent?.wrong || 0;
  }
  modalEmpty(): number {
    return this.emptyCount(this.modalStudent || {});
  }

  onExport() {
    window.location.href = `${BASE}/api/export_latest`;
  }
}
