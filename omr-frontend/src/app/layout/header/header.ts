
import { Component, HostListener, OnInit, OnDestroy, ChangeDetectorRef } from '@angular/core';
import { Router } from '@angular/router';
import { CommonModule } from '@angular/common';
import { Subscription } from 'rxjs';
import { AuthService } from '../../core/services/auth.service';
import { SidebarStateService } from '../../core/services/sidebar-state.service';

@Component({
  selector: 'app-header',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './header.html',
  styleUrl: './header.scss',
})

export class Header implements OnInit, OnDestroy {
  showProfilePopup = false;
  user: any = null;
  userLoaded = false;

  private sub!: Subscription;

  constructor(
    private router: Router,
    private authService: AuthService,
    private sidebarState: SidebarStateService,
    private cdr: ChangeDetectorRef,
  ) {}

  toggleSidebar() {
    this.sidebarState.toggle();
  }
  

  ngOnInit(): void {
    this.getCurrentUser();
  }

  ngOnDestroy(): void {
    this.sub?.unsubscribe();
  }

  getCurrentUser() {
    this.authService.getCurrentUser().subscribe({
      next: (res: any) => { this.user = res; this.userLoaded = true; },
      error: () => { this.userLoaded = true; },
    });
  }

  toggleProfilePopup() { this.showProfilePopup = !this.showProfilePopup; }

  logout() { this.authService.clearUserCache(); this.router.navigate(['/login']); }

  @HostListener('document:click', ['$event'])
  closePopup(event: Event) {
    if (!(event.target as HTMLElement).closest('.profile-wrapper')) {
      this.showProfilePopup = false;
    }
  }
}