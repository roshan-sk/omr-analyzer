import { Component, OnInit, ChangeDetectorRef } from '@angular/core';
import { RouterLink, RouterLinkActive } from '@angular/router';
import { CommonModule } from '@angular/common';
import { AuthService } from '../../core/services/auth.service';
import { Router } from '@angular/router';
import { SidebarStateService } from '../../core/services/sidebar-state.service';

@Component({
  selector: 'app-sidebar',
  standalone: true,
  imports: [CommonModule, RouterLink, RouterLinkActive],
  templateUrl: './sidebar.html',
  styleUrls: ['./sidebar.scss'],
})
export class Sidebar implements OnInit {
  username = '';
  role = '';
  isHoverOpen = false;

  get isSidebarOpen(): boolean {
    return this.sidebarState.isOpen;
  }

  get isVisuallyOpen(): boolean {
    return this.isSidebarOpen || this.isHoverOpen;
  }

  constructor(
    private authService: AuthService,
    private cdr: ChangeDetectorRef,
    private router: Router,
    private sidebarState: SidebarStateService,
  ) {}

  ngOnInit() {
    this.getCurrentUser();
  }

  toggleSidebar() {
    this.sidebarState.toggle();
    this.isHoverOpen = false;
  }

  onMouseEnter() {
    if (!this.isSidebarOpen) {
      this.isHoverOpen = true;
      this.cdr.detectChanges();
    }
  }

  onMouseLeave() {
    if (!this.isSidebarOpen) {
      this.isHoverOpen = false;
      this.cdr.detectChanges();
    }
  }

  getCurrentUser() {
    this.authService.getCurrentUser().subscribe({
      next: (response: any) => {
        this.username = response.username;
        this.role = response.role;
        this.cdr.detectChanges();
      },
      error: (error) => console.log(error),
    });
  }

  logout() {
    this.authService.clearUserCache();
    this.router.navigate(['/login']);
  }
}