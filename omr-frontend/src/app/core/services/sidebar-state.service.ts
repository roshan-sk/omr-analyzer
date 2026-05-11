import { Injectable } from '@angular/core';
import { BehaviorSubject } from 'rxjs';

@Injectable({ providedIn: 'root' })
export class SidebarStateService {
  private _isOpen = new BehaviorSubject<boolean>(true);
  isOpen$ = this._isOpen.asObservable();

  get isOpen(): boolean {
    return this._isOpen.value;
  }

  toggle() {
    this._isOpen.next(!this._isOpen.value);
  }
}