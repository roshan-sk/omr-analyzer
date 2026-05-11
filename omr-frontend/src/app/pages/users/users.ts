import { Component, OnInit, OnDestroy, ChangeDetectorRef } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { Subscription } from 'rxjs';
import { Sidebar } from '../../layout/sidebar/sidebar';
import { Header } from '../../layout/header/header';
import { UserService } from '../../core/services/user.service';
import { SidebarStateService } from '../../core/services/sidebar-state.service';
import { PaginationService } from '../../core/services/pagination.service';
import { PaginationComponent } from '../../shared/components/pagination/pagination';

@Component({
  selector: 'app-users',
  standalone: true,
  imports: [ CommonModule, FormsModule, Sidebar, Header, PaginationComponent],
  templateUrl: './users.html',
  styleUrls: ['./users.scss'],
})
export class Users implements OnInit, OnDestroy {
  users: any[] = [];
  filteredUsers: any[] = [];
  searchQuery = '';
  loading = false;
  errorMessage = '';
  isSidebarOpen = true;

  // ── Pagination ──────────────────────────────────────────────
  pageIndex = 0;
  pageSize = 5;

  get pagedUsers(): any[] {
    return this.paginationService.paginate(this.filteredUsers, this.pageIndex, this.pageSize);
  }

  get totalPages(): number {
    return this.paginationService.totalPages(this.filteredUsers.length, this.pageSize);
  }

  get rangeLabel(): string {
    return this.paginationService.rangeLabel(
      this.filteredUsers.length,
      this.pageIndex,
      this.pageSize,
      'users',
    );
  }

  onPageSizeChange(size: number) {
    this.pageSize = Number(size);
    this.pageIndex = 0;
    this.cdr.detectChanges();
  }
  // ────────────────────────────────────────────────────────────

  private sidebarSub!: Subscription;

  // Add modal
  showAddModal = false;
  addForm = { username: '', email: '', password: '', role: 'USER', is_active: true };
  addLoading = false;
  addError = '';
  showAddPassword = false;

  // Edit modal
  showEditModal = false;
  editingUser: any = null;
  editForm = { username: '', email: '', password: '', role: '', is_active: false };
  editLoading = false;
  editError = '';
  showEditPassword = false;

  // Delete modal
  showDeleteModal = false;
  deletingUser: any = null;
  deleteLoading = false;

  constructor(
    private userService: UserService,
    private cdr: ChangeDetectorRef,
    private sidebarState: SidebarStateService,
    private paginationService: PaginationService,
  ) {}

  ngOnInit(): void {
    this.isSidebarOpen = this.sidebarState.isOpen;
    this.sidebarSub = this.sidebarState.isOpen$.subscribe((open) => {
      this.isSidebarOpen = open;
      this.cdr.detectChanges();
    });
    this.getUsers();
  }

  ngOnDestroy(): void {
    this.sidebarSub?.unsubscribe();
  }

  getUsers() {
    this.loading = true;
    this.userService.getUsers().subscribe({
      next: (response: any) => {
        this.users = response;
        this.applySearch();
        this.loading = false;
        this.cdr.detectChanges();
      },
      error: (error) => {
        this.loading = false;
        this.errorMessage = error?.error?.msg || 'Failed to load users';
        this.cdr.detectChanges();
      },
    });
  }

  applySearch() {
    const q = this.searchQuery.trim().toLowerCase();
    this.filteredUsers = !q
      ? [...this.users]
      : this.users.filter(
          (u) =>
            u.username?.toLowerCase().includes(q) ||
            u.email?.toLowerCase().includes(q) ||
            u.role?.toLowerCase().includes(q),
        );
  }

  onSearchChange() {
    this.applySearch();
    this.pageIndex = 0;
  }

  // ── Add ────────────────────────────────────────────────────
  openAddModal() {
    this.addForm = { username: '', email: '', password: '', role: 'USER', is_active: true };
    this.addError = '';
    this.addLoading = false;
    this.showAddPassword = false;
    this.showAddModal = true;
  }

  closeAddModal() {
    this.showAddModal = false;
    this.addError = '';
    this.addLoading = false;
    this.showAddPassword = false;
  }

  saveNewUser() {
    this.addLoading = true;
    this.addError = '';
    this.userService.createUser({ ...this.addForm }).subscribe({
      next: () => {
        this.showAddModal = false;
        this.addLoading = false;
        this.pageIndex = 0;
        this.getUsers();
      },
      error: (e) => {
        this.addLoading = false;
        this.addError = e?.error?.msg || 'Failed to create user';
        this.cdr.detectChanges();
      },
    });
  }

  // ── Edit ───────────────────────────────────────────────────
  openEditModal(user: any) {
    this.editingUser = user;
    this.editForm = {
      username: user.username,
      email: user.email,
      password: '',
      role: user.role,
      is_active: user.is_active,
    };
    this.editError = '';
    this.editLoading = false;
    this.showEditPassword = false;
    this.showEditModal = true;
  }

  closeEditModal() {
    this.showEditModal = false;
    this.editingUser = null;
    this.editLoading = false;
    this.editError = '';
    this.showEditPassword = false;
  }

  saveUser() {
    this.editLoading = true;
    this.editError = '';
    const payload: any = {
      username: this.editForm.username,
      email: this.editForm.email,
      role: this.editForm.role,
      is_active: this.editForm.is_active,
    };
    if (this.editForm.password) payload.password = this.editForm.password;
    this.userService.updateUser(this.editingUser.id, payload).subscribe({
      next: () => {
        this.showEditModal = false;
        this.editingUser = null;
        this.editLoading = false;
        this.getUsers();
      },
      error: (e) => {
        this.editLoading = false;
        this.editError = e?.error?.msg || 'Failed to update user';
        this.cdr.detectChanges();
      },
    });
  }

  // ── Delete ─────────────────────────────────────────────────
  openDeleteModal(user: any) {
    this.deletingUser = user;
    this.deleteLoading = false;
    this.showDeleteModal = true;
  }

  closeDeleteModal() {
    this.showDeleteModal = false;
    this.deletingUser = null;
    this.deleteLoading = false;
  }

  confirmDelete() {
    this.deleteLoading = true;
    this.userService.deleteUser(this.deletingUser.id).subscribe({
      next: () => {
        this.users = this.users.filter((u) => u.id !== this.deletingUser.id);
        this.applySearch();
        // If deleting last item on current page, step back
        if (this.pageIndex >= this.totalPages) {
          this.pageIndex = Math.max(0, this.totalPages - 1);
        }
        this.showDeleteModal = false;
        this.deletingUser = null;
        this.deleteLoading = false;
        this.cdr.detectChanges();
      },
      error: (e) => {
        this.deleteLoading = false;
        this.errorMessage = e?.error?.msg || 'Failed to delete user';
        this.showDeleteModal = false;
        this.deletingUser = null;
        this.cdr.detectChanges();
      },
    });
  }
}
