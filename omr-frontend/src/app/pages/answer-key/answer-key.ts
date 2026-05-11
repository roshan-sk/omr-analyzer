import { Component, OnInit, OnDestroy, ChangeDetectorRef } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Sidebar } from '../../layout/sidebar/sidebar';
import { Header } from '../../layout/header/header';
import { AnswerKeyService } from '../../core/services/answer-key.service';
import { FormsModule } from '@angular/forms';
import { Subscription } from 'rxjs';
import { SidebarStateService } from '../../core/services/sidebar-state.service';

@Component({
  selector: 'app-answer-key',
  standalone: true,
  imports: [CommonModule, Sidebar, Header, FormsModule],
  templateUrl: './answer-key.html',
  styleUrls: ['./answer-key.scss'],
})
export class AnswerKey implements OnInit, OnDestroy {

  levels = ['lower_primary', 'upper_primary', 'junior', 'intermediate', 'senior', 'open'];
  selectedLevel = 'intermediate';
  options = ['A', 'B', 'C', 'D', 'E'];

  loading = false;
  saving  = false;
  isSidebarOpen = true;

  private sidebarSub!: Subscription;

  keyData:  Record<string, string> = {};
  editData: Record<string, string> = {};
  ranges: any[] = [];

  questions = Array.from({ length: 40 }, (_, i) => {
    const n = String(i + 1).padStart(2, '0');
    return { key: `Q${n}`, label: `Q${n}` };
  });

  constructor(
    private answerKeyService: AnswerKeyService,
    private cdr: ChangeDetectorRef,
    private sidebarState: SidebarStateService,
  ) {
    this.isSidebarOpen = this.sidebarState.isOpen;
  }

  ngOnInit(): void {
    this.sidebarSub = this.sidebarState.isOpen$.subscribe(open => {
      this.isSidebarOpen = open;
      this.cdr.detectChanges();
    });
    this.loadLevel(this.selectedLevel);
  }

  ngOnDestroy(): void {
    this.sidebarSub?.unsubscribe();
  }

  selectLevel(level: string) {
    if (this.selectedLevel === level) return;
    this.selectedLevel = level;
    this.loadLevel(level);
  }

  loadLevel(level: string) {
    this.loading  = true;
    this.keyData  = {};
    this.editData = {};
    this.ranges   = [];
    this.cdr.detectChanges();

    this.answerKeyService.getAnswerKey(level).subscribe({
      next: (response: any) => {
        this.keyData  = response.answers      || {};
        this.editData = { ...this.keyData };
        this.ranges   = response.scoring_rules || [];
        this.loading  = false;
        this.cdr.detectChanges();
      },
      error: (err) => {
        console.error('Load error:', err);
        this.loading = false;
        this.cdr.detectChanges();
      },
    });
  }

  selectOption(qKey: string, opt: string) {
    const updated = { ...this.editData };
    if (updated[qKey] === opt) { delete updated[qKey]; }
    else { updated[qKey] = opt; }
    this.editData = updated;
    this.cdr.detectChanges();
  }

  cancelEdit() {
    this.editData = { ...this.keyData };
    this.cdr.detectChanges();
  }

  saveAnswerKey() {
    if (this.saving) return;
    this.saving = true;
    this.cdr.detectChanges();

    const payload = {
      level:         this.selectedLevel,
      answers:       this.editData,
      scoring_rules: this.ranges,
    };

    this.answerKeyService.saveAnswerKey(payload).subscribe({
      next: () => {
        this.keyData = { ...this.editData };
        this.saving  = false;
        this.cdr.detectChanges();
      },
      error: (err) => {
        console.error('Save error:', err);
        this.saving = false;
        this.cdr.detectChanges();
      },
    });
  }

  addRange() {
    this.ranges = [...this.ranges, { from: 1, to: 40, correct: 1, wrong: 0, empty: 0 }];
    this.cdr.detectChanges();
  }

  removeRange(i: number) {
    this.ranges = this.ranges.filter((_, idx) => idx !== i);
    this.cdr.detectChanges();
  }

  formatLevel(l: string) { return l.replace(/_/g, ' '); }

  get setCount() { return Object.keys(this.editData).length; }
  get isDirty()  { return JSON.stringify(this.editData) !== JSON.stringify(this.keyData); }
}