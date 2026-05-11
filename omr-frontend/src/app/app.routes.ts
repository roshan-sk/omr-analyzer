import { Routes } from '@angular/router';

import { Analyzer } from './pages/analyzer/analyzer';
import { Login } from './pages/login/login';
import { Users } from './pages/users/users';
import { AnswerKey } from './pages/answer-key/answer-key';

export const routes: Routes = [
  {
    path: '',
    redirectTo: 'login',
    pathMatch: 'full',
  },

  {
    path: 'login',
    title: 'Login — OMR Analyzer',
    component: Login,
  },

  {
    path: 'analyzer',
    title: 'Analyzer — OMR Analyzer',
    component: Analyzer,
  },

  {
    path: 'answer-key',
    title: 'Answer Key — OMR Analyzer',
    component: AnswerKey,
  },

  {
    path: 'users',
    title: 'Users — OMR Analyzer',
    component: Users,
  },
];